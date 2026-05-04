from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vjepa_embedding_utils import ensure_embedding_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run linear probes on V-JEPA ECG segment embeddings."
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
        default=Path("src/data/vjepa_linear_probe"),
        help="Where to save metrics, plots, and reports.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("src/data/vjepa_embedding_experiments/records"),
        help="Embedding cache directory. One .npz file per record is expected unless a legacy .npz path is provided.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/vjepa2-vitl-fpc64-256",
        help="Hugging Face V-JEPA model name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used if embeddings need to be recomputed.",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Optional subset of exported records to use, e.g. --records 100 102 104.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Optional limit on the number of segments to evaluate.",
    )
    parser.add_argument(
        "--target-num-frames",
        type=int,
        default=16,
        help="Fixed number of frames per V-JEPA clip.",
    )
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=8,
        help="Stride when splitting long segments into multiple clips.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of clips per V-JEPA forward pass if embeddings must be recomputed.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached embeddings and recompute them.",
    )
    return parser.parse_args()


def build_probe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def summarize_cv(metrics: dict[str, np.ndarray]) -> dict:
    return {
        "accuracy_mean": float(np.mean(metrics["test_accuracy"])),
        "accuracy_std": float(np.std(metrics["test_accuracy"])),
        "f1_mean": float(np.mean(metrics["test_f1"])),
        "f1_std": float(np.std(metrics["test_f1"])),
        "roc_auc_mean": float(np.mean(metrics["test_roc_auc"])),
        "roc_auc_std": float(np.std(metrics["test_roc_auc"])),
    }


def run_stratified_cv(features: np.ndarray, labels: np.ndarray) -> dict:
    min_class_count = int(np.bincount(labels).min())
    n_splits = min(5, min_class_count)
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for stratified cross-validation.")

    probe = build_probe()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = cross_validate(
        probe,
        features,
        labels,
        cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=False,
    )
    return summarize_cv(metrics)


def run_leave_one_record_out(
    features: np.ndarray,
    labels: np.ndarray,
    record_names: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    probe = build_probe()
    logo = LeaveOneGroupOut()
    fold_rows: list[dict] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for fold_index, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups=record_names), start=1):
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        probe.fit(features[train_idx], y_train)
        y_pred = probe.predict(features[test_idx])
        y_score = probe.predict_proba(features[test_idx])[:, 1]

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        held_out_record = str(record_names[test_idx][0])
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_record": held_out_record,
                "num_segments": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_score)),
            }
        )

    if not fold_rows:
        raise ValueError(
            "Leave-one-record-out evaluation could not be run because each held-out record "
            "must contain both classes and each training split must also contain both classes."
        )

    return fold_rows, np.concatenate(y_true_all), np.concatenate(y_pred_all)


def save_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_true_label_pca(features: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    centered = StandardScaler().fit_transform(features)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T
    plt.figure(figsize=(7, 6))
    for label_value, color, marker, name in [
        (1, "tab:red", "o", "not-normal rhythm"),
        (0, "tab:blue", "x", "normal rhythm"),
    ]:
        mask = labels == label_value
        plt.scatter(projection[mask, 0], projection[mask, 1], c=color, marker=marker, s=60, alpha=0.8, label=name)
    plt.title("Segment embeddings colored by true rhythm label")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_records = set(args.records) if args.records else None
    examples, embeddings, labels, record_names = ensure_embedding_cache(
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
        raise ValueError("Linear probe evaluation requires both normal and not-normal rhythm labels in the selected data.")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    stratified_summary = run_stratified_cv(embeddings, labels)
    fold_rows, y_true, y_pred = run_leave_one_record_out(embeddings, labels, record_names)

    confusion = confusion_matrix(y_true, y_pred)
    summary = {
        "num_segments": int(len(examples)),
        "num_records": int(len(np.unique(record_names))),
        "embedding_dim": int(embeddings.shape[1]),
        "probe_input": "raw_vjepa_embeddings",
        "stratified_cv": stratified_summary,
        "leave_one_record_out": {
            "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_rows])),
            "accuracy_std": float(np.std([row["accuracy"] for row in fold_rows])),
            "f1_mean": float(np.mean([row["f1"] for row in fold_rows])),
            "f1_std": float(np.std([row["f1"] for row in fold_rows])),
            "roc_auc_mean": float(np.mean([row["roc_auc"] for row in fold_rows])),
            "roc_auc_std": float(np.std([row["roc_auc"] for row in fold_rows])),
        },
        "confusion_matrix": confusion.tolist(),
    }

    save_rows_csv(args.output_dir / "leave_one_record_out_results.csv", fold_rows)
    (args.output_dir / "linear_probe_summary.json").write_text(json.dumps(summary, indent=2))
    plot_true_label_pca(embeddings, labels, args.output_dir / "true_labels_pca.png")

    print("\nStratified CV")
    print(
        f"accuracy={stratified_summary['accuracy_mean']:.3f} +/- {stratified_summary['accuracy_std']:.3f}, "
        f"f1={stratified_summary['f1_mean']:.3f} +/- {stratified_summary['f1_std']:.3f}, "
        f"roc_auc={stratified_summary['roc_auc_mean']:.3f} +/- {stratified_summary['roc_auc_std']:.3f}"
    )

    print("\nLeave-one-record-out")
    for row in fold_rows:
        print(
            f"record={row['held_out_record']} "
            f"accuracy={row['accuracy']:.3f} "
            f"precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} "
            f"roc_auc={row['roc_auc']:.3f}"
        )

    print("\nConfusion matrix")
    print(confusion)
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
