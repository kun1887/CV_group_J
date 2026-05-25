from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from vjepa_embedding_utils import ensure_clip_embedding_cache, ensure_embedding_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed exported ECG rhythm segments with V-JEPA and compare clustering methods."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/data/mitdb_vjepa_frames"),
        help="Root folder containing per-record segment exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data/vjepa_embedding_experiments"),
        help="Where to save embeddings, metrics, and PCA plots.",
    )
    parser.add_argument("--model-name", default="facebook/vjepa2-vitl-fpc64-256", help="Hugging Face V-JEPA model name.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device used for embedding extraction.")
    parser.add_argument("--records", nargs="+", help="Optional subset of exported records to use, e.g. --records 100 102 104.")
    parser.add_argument("--max-segments", type=int, help="Optional limit on the number of segments to embed.")
    parser.add_argument("--target-num-frames", type=int, default=16, help="Fixed number of frames per V-JEPA clip.")
    parser.add_argument("--clip-stride", type=int, default=8, help="Stride when splitting long segments into multiple clips.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of clips per V-JEPA forward pass.")
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=32,
        help="Number of PCA dimensions used before clustering.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("src/data/vjepa_embedding_experiments/records"),
        help=(
            "Embedding cache directory. In default mode this is a pooled embedding cache. "
            "With --pool-record-clips, this is a per-record clip embedding cache."
        ),
    )
    parser.add_argument(
        "--pool-record-clips",
        action="store_true",
        help="Load clip embeddings and pool all clips from each record into one record embedding before clustering.",
    )
    parser.add_argument(
        "--clip-pooling",
        choices=("mean", "max"),
        default="mean",
        help="Pooling operation used with --pool-record-clips.",
    )
    parser.add_argument(
        "--clip-value",
        type=float,
        default=1e6,
        help="Absolute value used to clip embedding features before scaling and PCA.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PCA scatter plot generation.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached embeddings and recompute them.",
    )
    return parser.parse_args()


def pool_clip_embeddings_by_record(
    clip_embeddings: np.ndarray,
    clip_meta: dict[str, np.ndarray],
    pooling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record_names = clip_meta["clip_record_names"].astype(str)
    labels = clip_meta["clip_labels"].astype(np.int32)
    pooled_embeddings: list[np.ndarray] = []
    pooled_labels: list[int] = []
    pooled_record_names: list[str] = []

    for record_name in sorted(set(record_names), key=lambda value: int(value) if value.isdigit() else value):
        mask = record_names == record_name
        record_embeddings = clip_embeddings[mask]
        record_labels = np.unique(labels[mask])
        if len(record_labels) != 1:
            raise ValueError(f"Record {record_name} has inconsistent clip labels: {record_labels.tolist()}")

        record_embeddings = np.nan_to_num(record_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        if pooling == "mean":
            pooled = record_embeddings.mean(axis=0, dtype=np.float64)
        elif pooling == "max":
            pooled = record_embeddings.max(axis=0)
        else:
            raise ValueError(f"Unsupported clip pooling method: {pooling}")

        pooled_embeddings.append(pooled)
        pooled_labels.append(int(record_labels[0]))
        pooled_record_names.append(record_name)

    return (
        np.stack(pooled_embeddings).astype(np.float32),
        np.array(pooled_labels, dtype=np.int32),
        np.array(pooled_record_names),
    )


def cluster_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    mask = cluster_labels >= 0
    if not np.any(mask):
        return float("nan")

    filtered_true = true_labels[mask]
    filtered_cluster = cluster_labels[mask]
    total = len(filtered_true)
    purity_total = 0
    for cluster_id in np.unique(filtered_cluster):
        cluster_mask = filtered_cluster == cluster_id
        cluster_truth = filtered_true[cluster_mask]
        counts = np.bincount(cluster_truth)
        purity_total += int(np.max(counts))
    return purity_total / total


def evaluate_clustering(name: str, predicted_clusters: np.ndarray, features: np.ndarray, true_labels: np.ndarray) -> dict:
    unique_clusters = np.unique(predicted_clusters)
    non_noise_clusters = unique_clusters[unique_clusters >= 0]
    usable_for_silhouette = len(non_noise_clusters) >= 2 and np.sum(predicted_clusters >= 0) >= len(non_noise_clusters)

    metrics = {
        "method": name,
        "num_clusters_found": int(len(non_noise_clusters)),
        "num_noise_points": int(np.sum(predicted_clusters < 0)),
        "ari": float(adjusted_rand_score(true_labels, predicted_clusters)),
        "nmi": float(normalized_mutual_info_score(true_labels, predicted_clusters)),
        "purity": float(cluster_purity(true_labels, predicted_clusters)),
        "silhouette": None,
    }

    if usable_for_silhouette:
        mask = predicted_clusters >= 0
        metrics["silhouette"] = float(silhouette_score(features[mask], predicted_clusters[mask]))

    return metrics


def run_clustering_suite(features: np.ndarray, true_labels: np.ndarray, random_state: int = 42) -> tuple[list[dict], dict[str, np.ndarray]]:
    n_samples = features.shape[0]
    spectral_neighbors = max(2, min(10, n_samples - 1))
    clusterers = {
        "kmeans": KMeans(n_clusters=2, n_init=20, random_state=random_state),
        "gaussian_mixture": GaussianMixture(n_components=2, covariance_type="full", random_state=random_state),
        "agglomerative_ward": AgglomerativeClustering(n_clusters=2, linkage="ward"),
        "agglomerative_average": AgglomerativeClustering(n_clusters=2, linkage="average"),
        "spectral": SpectralClustering(
            n_clusters=2,
            assign_labels="kmeans",
            random_state=random_state,
            affinity="nearest_neighbors",
            n_neighbors=spectral_neighbors,
        ),
        "birch": Birch(n_clusters=2),
        "dbscan": DBSCAN(eps=2.2, min_samples=4),
    }

    results: list[dict] = []
    assignments: dict[str, np.ndarray] = {}
    for name, clusterer in clusterers.items():
        if hasattr(clusterer, "fit_predict"):
            predicted = clusterer.fit_predict(features)
        else:
            clusterer.fit(features)
            predicted = clusterer.predict(features)

        predicted = np.asarray(predicted, dtype=np.int32)
        assignments[name] = predicted
        results.append(evaluate_clustering(name, predicted, features, true_labels))

    results.sort(key=lambda row: (row["ari"], row["nmi"], row["purity"]), reverse=True)
    return results, assignments


def save_results_table(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        header = "method,num_clusters_found,num_noise_points,ari,nmi,purity,silhouette\n"
        handle.write(header)
        for row in results:
            silhouette = "" if row["silhouette"] is None else f"{row['silhouette']:.6f}"
            handle.write(
                f"{row['method']},{row['num_clusters_found']},{row['num_noise_points']},"
                f"{row['ari']:.6f},{row['nmi']:.6f},{row['purity']:.6f},{silhouette}\n"
            )


def plot_assignments(
    projection: np.ndarray,
    true_labels: np.ndarray,
    assignments: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    label_names = np.where(true_labels == 0, "normal rhythm", "not-normal rhythm")
    for label_value, color, marker in [(1, "tab:red", "o"), (0, "tab:blue", "x")]:
        mask = true_labels == label_value
        if not np.any(mask):
            continue
        plt.scatter(projection[mask, 0], projection[mask, 1], color=color, marker=marker, s=55, alpha=0.75, label=label_names[mask][0])
    plt.title("Segment embeddings colored by true rhythm label")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "true_labels_pca.png", dpi=180)
    plt.close()

    for method, predicted in assignments.items():
        plt.figure(figsize=(7, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=predicted, cmap="tab10", s=45, alpha=0.8)
        plt.title(f"Segment embeddings clustered by {method}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / f"{method}_pca.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_records = set(args.records) if args.records else None
    if args.pool_record_clips:
        _, clip_embeddings, clip_meta = ensure_clip_embedding_cache(
            dataset_root=args.dataset_root,
            embedding_cache=args.embedding_cache,
            selected_records=selected_records,
            max_segments=args.max_segments,
            model_name=args.model_name,
            device_arg=args.device,
            batch_size=args.batch_size,
            target_num_frames=args.target_num_frames,
            clip_stride=args.clip_stride,
            force_recompute=args.force_recompute,
        )
        embeddings, labels, record_names = pool_clip_embeddings_by_record(
            clip_embeddings,
            clip_meta,
            args.clip_pooling,
        )
        print(
            f"Pooled {len(clip_embeddings)} clip embedding(s) into "
            f"{len(record_names)} record embedding(s) using {args.clip_pooling} pooling."
        )
    else:
        _, embeddings, labels, _ = ensure_embedding_cache(
            dataset_root=args.dataset_root,
            embedding_cache=args.embedding_cache,
            selected_records=selected_records,
            max_segments=args.max_segments,
            model_name=args.model_name,
            device_arg=args.device,
            batch_size=args.batch_size,
            target_num_frames=args.target_num_frames,
            clip_stride=args.clip_stride,
            force_recompute=args.force_recompute,
        )
    if len(np.unique(labels)) < 2:
        raise ValueError("Clustering requires both normal and not-normal rhythm labels in the selected data.")

    scaler = StandardScaler()
    embeddings = np.nan_to_num(embeddings.astype(np.float64), nan=0.0, posinf=args.clip_value, neginf=-args.clip_value)
    embeddings = np.clip(embeddings, -args.clip_value, args.clip_value)
    scaled_embeddings = scaler.fit_transform(embeddings)
    scaled_embeddings = np.nan_to_num(scaled_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    pca_dims = min(args.pca_dims, scaled_embeddings.shape[0], scaled_embeddings.shape[1])
    reduced_embeddings = PCA(n_components=pca_dims, random_state=42, svd_solver="full").fit_transform(scaled_embeddings)
    projection_2d = PCA(n_components=2, random_state=42, svd_solver="full").fit_transform(scaled_embeddings)

    results, assignments = run_clustering_suite(reduced_embeddings, labels)
    save_results_table(results, args.output_dir / "clustering_results.csv")
    if not args.no_plots:
        plot_assignments(projection_2d, labels, assignments, args.output_dir)

    print("\nClustering results")
    for row in results:
        silhouette = "n/a" if row["silhouette"] is None else f"{row['silhouette']:.3f}"
        print(
            f"{row['method']}: "
            f"ARI={row['ari']:.3f}, "
            f"NMI={row['nmi']:.3f}, "
            f"purity={row['purity']:.3f}, "
            f"silhouette={silhouette}, "
            f"clusters={row['num_clusters_found']}, "
            f"noise={row['num_noise_points']}"
        )

    best = results[0]
    print(
        "\nBest method by ARI/NMI/purity ordering: "
        f"{best['method']} "
        f"(ARI={best['ari']:.3f}, NMI={best['nmi']:.3f}, purity={best['purity']:.3f})"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
