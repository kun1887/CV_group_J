from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression probe on pooled cached V-JEPA clip embeddings using PTB-XL folds."
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("src/data/ptbxl_vjepa_clip_embedding_experiments_fpc16/records"),
        help="Directory containing per-record clip embedding .npz files.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/data/ptbxl_vjepa_frames"),
        help="PTB-XL frame dataset root used to read labels and strat_fold from metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data/ptbxl_vjepa_linear_probe_fpc16"),
        help="Where to save summary and prediction CSVs.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "max"),
        default="mean",
        help="How to pool clip embeddings into one record embedding.",
    )
    parser.add_argument(
        "--clip-value",
        type=float,
        default=1e6,
        help="Absolute value used to clip pooled embedding features before fitting.",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Do not tune the decision threshold on validation F1; still reports default-threshold results.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def pool_clip_embeddings(clip_embeddings: np.ndarray, pooling: str) -> np.ndarray:
    clip_embeddings = np.nan_to_num(clip_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    if pooling == "mean":
        return clip_embeddings.mean(axis=0, dtype=np.float64)
    if pooling == "max":
        return clip_embeddings.max(axis=0)
    raise ValueError(f"Unsupported pooling method: {pooling}")


def load_dataset(
    embedding_cache: Path,
    dataset_root: Path,
    pooling: str,
    clip_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_names: list[str] = []
    folds: list[int] = []
    labels: list[int] = []
    embeddings: list[np.ndarray] = []

    for cache_path in sorted(embedding_cache.glob("*.npz"), key=natural_key):
        record_name = cache_path.stem
        metadata_path = dataset_root / record_name / "segment_0000" / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        if "strat_fold" not in metadata:
            continue

        cache = np.load(cache_path, allow_pickle=False)
        if "embeddings" not in cache:
            raise ValueError(f"Missing embeddings array in {cache_path}")

        label = 0 if metadata["class"] == "normal rhythm" else 1
        embeddings.append(pool_clip_embeddings(cache["embeddings"], pooling))
        labels.append(label)
        folds.append(int(metadata["strat_fold"]))
        record_names.append(record_name)

    if not embeddings:
        raise ValueError(f"No usable cached clip embeddings found in {embedding_cache}.")

    features = np.stack(embeddings).astype(np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=clip_value, neginf=-clip_value)
    features = np.clip(features, -clip_value, clip_value)
    return (
        np.array(record_names),
        np.array(folds, dtype=np.int32),
        np.array(labels, dtype=np.int32),
        features,
    )


def build_probe() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
    )


def save_model_parameters(path: Path, probe: object, threshold: float) -> None:
    scaler = probe.named_steps["standardscaler"]
    classifier = probe.named_steps["logisticregression"]
    scaled_weight = classifier.coef_[0].astype(np.float64)
    scaled_intercept = float(classifier.intercept_[0])
    scaler_mean = scaler.mean_.astype(np.float64)
    scaler_scale = scaler.scale_.astype(np.float64)

    raw_weight = scaled_weight / scaler_scale
    raw_intercept = scaled_intercept - float(np.sum(scaled_weight * scaler_mean / scaler_scale))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        scaled_weight=scaled_weight,
        scaled_intercept=np.array(scaled_intercept, dtype=np.float64),
        raw_weight=raw_weight,
        raw_intercept=np.array(raw_intercept, dtype=np.float64),
        threshold=np.array(float(threshold), dtype=np.float64),
        class_names=np.array(["normal rhythm", "not-normal rhythm"]),
    )


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(np.int32)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.unique(scores):
        f1 = f1_score(labels, (scores >= threshold).astype(np.int32), zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and float(threshold) < best_threshold):
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def save_predictions(
    path: Path,
    record_names: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (scores >= threshold).astype(np.int32)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_name", "true_label", "predicted_label", "score_not_normal"])
        writer.writeheader()
        for record_name, label, prediction, score in zip(record_names, labels, predictions, scores):
            writer.writerow(
                {
                    "record_name": str(record_name),
                    "true_label": int(label),
                    "predicted_label": int(prediction),
                    "score_not_normal": float(score),
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_names, folds, labels, features = load_dataset(
        args.embedding_cache,
        args.dataset_root,
        args.pooling,
        args.clip_value,
    )
    train_mask = np.isin(folds, [1, 2, 3, 4, 5, 6, 7, 8])
    val_mask = folds == 9
    test_mask = folds == 10
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Expected PTB-XL strat_fold train/validation/test records.")

    probe = build_probe()
    probe.fit(features[train_mask], labels[train_mask])
    val_scores = probe.predict_proba(features[val_mask])[:, 1]
    test_scores = probe.predict_proba(features[test_mask])[:, 1]

    tuned_threshold = 0.5 if args.no_threshold_tuning else select_f1_threshold(labels[val_mask], val_scores)
    summary = {
        "embedding_cache": str(args.embedding_cache),
        "dataset_root": str(args.dataset_root),
        "pooling": args.pooling,
        "num_records": int(len(record_names)),
        "split_counts": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "label_counts": {
            "train": np.bincount(labels[train_mask], minlength=2).astype(int).tolist(),
            "validation": np.bincount(labels[val_mask], minlength=2).astype(int).tolist(),
            "test": np.bincount(labels[test_mask], minlength=2).astype(int).tolist(),
        },
        "default_threshold": {
            "threshold": 0.5,
            "validation": compute_metrics(labels[val_mask], val_scores, 0.5),
            "test": compute_metrics(labels[test_mask], test_scores, 0.5),
        },
        "max_f1_validation_threshold": {
            "threshold": float(tuned_threshold),
            "validation": compute_metrics(labels[val_mask], val_scores, tuned_threshold),
            "test": compute_metrics(labels[test_mask], test_scores, tuned_threshold),
        },
    }

    (args.output_dir / "linear_probe_summary.json").write_text(json.dumps(summary, indent=2))
    save_model_parameters(args.output_dir / "linear_probe_parameters.npz", probe, tuned_threshold)
    save_predictions(
        args.output_dir / "validation_predictions.csv",
        record_names[val_mask],
        labels[val_mask],
        val_scores,
        tuned_threshold,
    )
    save_predictions(
        args.output_dir / "held_out_predictions.csv",
        record_names[test_mask],
        labels[test_mask],
        test_scores,
        tuned_threshold,
    )

    test_metrics = summary["max_f1_validation_threshold"]["test"]
    print("Linear probe on pooled clip embeddings")
    print(f"records={len(record_names)} train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    print(f"threshold={tuned_threshold:.6f}")
    print(
        "test metrics: "
        f"roc_auc={test_metrics['roc_auc']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f}"
    )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
