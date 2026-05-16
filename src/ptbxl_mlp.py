from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_FOLDS = set(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP on PTB-XL V-JEPA embeddings using official folds.")
    parser.add_argument("--embedding-cache", type=Path, default=Path("src/data/ptbxl_vjepa_embeddings/records"))
    parser.add_argument("--metadata", type=Path, default=Path("src/data/ptbxl/ptbxl_database.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/data/ptbxl_results/mlp"))
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1024, 512])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_fold_map(metadata_path: Path) -> dict[str, int]:
    fold_map: dict[str, int] = {}
    with metadata_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fold_map[str(int(float(row["ecg_id"])))] = int(float(row["strat_fold"]))
    return fold_map


def load_embeddings(cache_dir: Path, fold_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings, labels, folds = [], [], []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        record_id = npz_path.stem
        if record_id not in fold_map:
            continue
        data = np.load(npz_path)
        embeddings.append(data["embeddings"].astype(np.float32).reshape(-1))
        labels.append(int(data["labels"][0]))
        folds.append(fold_map[record_id])
    return np.array(embeddings), np.array(labels), np.array(folds)


def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers += [nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)]
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, 2))
    return nn.Sequential(*layers)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate(model: nn.Sequential, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_labels.extend(y_batch.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    fold_map = load_fold_map(args.metadata)
    embeddings, labels, folds = load_embeddings(args.embedding_cache, fold_map)
    print(f"Loaded {len(embeddings)} records. Device: {device}")

    train_mask = np.isin(folds, list(TRAIN_FOLDS))
    val_mask = folds == VAL_FOLD
    test_mask = folds == TEST_FOLD

    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_mask]).astype(np.float32)
    X_val = scaler.transform(embeddings[val_mask]).astype(np.float32)
    X_test = scaler.transform(embeddings[test_mask]).astype(np.float32)
    y_train, y_val, y_test = labels[train_mask], labels[val_mask], labels[test_mask]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long()),
        batch_size=args.batch_size,
    )

    counts = np.bincount(y_train).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / counts, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_mlp(X_train.shape[1], args.hidden_dims, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state, patience_counter = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

        val_labels, val_preds, val_scores = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        print(f"epoch={epoch:03d} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    val_labels, val_preds, val_scores = evaluate(model, val_loader, device)
    test_labels, test_preds, test_scores = evaluate(model, test_loader, device)

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold, best_threshold_f1 = 0.5, -1.0
    threshold_sweep = []
    for t in thresholds:
        preds = (val_scores >= t).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        threshold_sweep.append({"threshold": float(t), "val_f1": float(f1)})
        if f1 > best_threshold_f1:
            best_threshold_f1 = f1
            best_threshold = float(t)

    print(f"Best threshold on val: {best_threshold:.2f} (val_f1={best_threshold_f1:.4f})")
    tuned_val_pred = (val_scores >= best_threshold).astype(int)
    tuned_test_pred = (test_scores >= best_threshold).astype(int)

    summary = {
        "classifier": "mlp",
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "val": compute_metrics(val_labels, val_preds, val_scores),
        "test": compute_metrics(test_labels, test_preds, test_scores),
        "best_threshold": best_threshold,
        "val_tuned": compute_metrics(val_labels, tuned_val_pred, val_scores),
        "test_tuned": compute_metrics(test_labels, tuned_test_pred, test_scores),
        "threshold_sweep": threshold_sweep,
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nVal metrics (default threshold=0.5):")
    print(f"  accuracy={summary['val']['accuracy']:.3f} f1={summary['val']['f1']:.3f} recall={summary['val']['recall']:.3f} roc_auc={summary['val']['roc_auc']:.3f}")
    print(f"Val metrics (tuned threshold={best_threshold:.2f}):")
    print(f"  accuracy={summary['val_tuned']['accuracy']:.3f} f1={summary['val_tuned']['f1']:.3f} recall={summary['val_tuned']['recall']:.3f} roc_auc={summary['val_tuned']['roc_auc']:.3f}")
    print("\nTest metrics (default threshold=0.5):")
    print(f"  accuracy={summary['test']['accuracy']:.3f} f1={summary['test']['f1']:.3f} recall={summary['test']['recall']:.3f} roc_auc={summary['test']['roc_auc']:.3f}")
    print(f"Test metrics (tuned threshold={best_threshold:.2f}):")
    print(f"  accuracy={summary['test_tuned']['accuracy']:.3f} f1={summary['test_tuned']['f1']:.3f} recall={summary['test_tuned']['recall']:.3f} roc_auc={summary['test_tuned']['roc_auc']:.3f}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
