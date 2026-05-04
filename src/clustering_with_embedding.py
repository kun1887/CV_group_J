from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
from vjepa_embedding_utils import ensure_embedding_cache


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
        help="Embedding cache directory. One .npz file per record is expected unless a legacy .npz path is provided.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached embeddings and recompute them.",
    )
    return parser.parse_args()


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
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    label_names = np.where(true_labels == 0, "normal rhythm", "not-normal rhythm")
    for label_value, color, marker in [(1, "tab:red", "o"), (0, "tab:blue", "x")]:
        mask = true_labels == label_value
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
    scaled_embeddings = scaler.fit_transform(embeddings)
    pca_dims = min(args.pca_dims, scaled_embeddings.shape[0], scaled_embeddings.shape[1])
    reduced_embeddings = PCA(n_components=pca_dims, random_state=42).fit_transform(scaled_embeddings)
    projection_2d = PCA(n_components=2, random_state=42).fit_transform(scaled_embeddings)

    results, assignments = run_clustering_suite(reduced_embeddings, labels)
    save_results_table(results, args.output_dir / "clustering_results.csv")
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
