from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


TRAIN_FOLDS = set(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10
FS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer on raw PTB-XL ECG signal as V-JEPA baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("src/data/ptbxl"))
    parser.add_argument("--cache-root", type=Path, default=Path("src/data/t_caches"))
    parser.add_argument("--window-sec", type=float, default=1.0, help="Token window size in seconds. Must match an existing cache.")
    parser.add_argument("--overlap", type=float, default=0.0, help="Fractional overlap used when building the cache (e.g. 0.5). Must match the cache.")
    parser.add_argument("--leads", nargs="+", type=int, default=[0], help="Lead indices to use (e.g. --leads 0 1 2). Each must have a matching cache.")
    parser.add_argument("--output-dir", type=Path, default=Path("src/data/ptbxl_results/transformer"))
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
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


def load_metadata(data_dir: Path) -> list[dict]:
    rows = []
    with (data_dir / "ptbxl_database.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["ecg_id"] = str(int(float(row["ecg_id"])))
            row["strat_fold"] = int(float(row["strat_fold"]))
            row["scp_codes_dict"] = ast.literal_eval(row["scp_codes"])
            rows.append(row)
    return rows


def load_signal(data_dir: Path, filename_lr: str) -> np.ndarray | None:
    try:
        import wfdb
        rec = wfdb.rdrecord(str(data_dir / filename_lr))
        return rec.p_signal.astype(np.float32)  # (1000, 12)
    except Exception:
        return None


def signal_to_tokens(signal: np.ndarray, lead: int, window_samples: int) -> np.ndarray:
    lead_signal = signal[:, lead]
    num_tokens = len(lead_signal) // window_samples
    return lead_signal[:num_tokens * window_samples].reshape(num_tokens, window_samples).astype(np.float32)


def load_cache(cache_root: Path, window_sec: float, lead: int, overlap: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    ov_tag = f"_ov{int(round(overlap * 100))}" if overlap > 0 else ""
    cache_path = cache_root / f"{window_sec:.2f}s{ov_tag}_lead{lead}" / "cache.npz"
    if not cache_path.exists():
        return None
    print(f"  Loading cache: {cache_path}")
    cache = np.load(cache_path)
    return cache["tokens"], cache["labels"], cache["folds"]


def load_all_leads(
    cache_root: Path,
    window_sec: float,
    leads: list[int],
    data_dir: Path,
    overlap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_per_lead = []
    labels_ref, folds_ref = None, None

    for lead in leads:
        result = load_cache(cache_root, window_sec, lead, overlap)
        if result is None:
            ov_str = f" --overlap {overlap}" if overlap > 0 else ""
            raise FileNotFoundError(
                f"No cache found for lead {lead} at window {window_sec}s overlap {overlap}. "
                f"Run: python src/build_ecg_token_cache.py --window-sec {window_sec}{ov_str} --lead {lead}"
            )
        tokens, labels, folds = result
        tokens_per_lead.append(tokens)
        if labels_ref is None:
            labels_ref, folds_ref = labels, folds

    # Concatenate leads along the token feature dimension: (N, T, W*num_leads)
    all_tokens = np.concatenate(tokens_per_lead, axis=2)
    return all_tokens, labels_ref, folds_ref


def sinusoidal_positional_encoding(num_positions: int, embed_dim: int) -> torch.Tensor:
    pe = torch.zeros(num_positions, embed_dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ECGTransformer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, num_layers: int, dropout: float, num_tokens: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_buffer("pos_enc", sinusoidal_positional_encoding(num_tokens + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_enc.unsqueeze(0)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)


class ECGDataset(Dataset):
    def __init__(self, tokens: np.ndarray, labels: np.ndarray) -> None:
        self.tokens = torch.from_numpy(tokens).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


def evaluate(model: ECGTransformer, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_labels.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def tune_threshold(val_labels: np.ndarray, val_scores: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(val_labels, (val_scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    window_samples = int(args.window_sec * FS)
    input_dim = window_samples * len(args.leads)

    print(f"Device: {device}")
    print(f"Window: {args.window_sec}s ({window_samples} samples) | Leads: {args.leads} | Input dim per token: {input_dim}")

    print("Loading token caches...")
    all_tokens, all_labels, all_folds = load_all_leads(args.cache_root, args.window_sec, args.leads, args.data_dir, args.overlap)

    train_mask = np.isin(all_folds, list(TRAIN_FOLDS))
    val_mask = all_folds == VAL_FOLD
    test_mask = all_folds == TEST_FOLD

    X_train, y_train = all_tokens[train_mask], all_labels[train_mask]
    X_val, y_val = all_tokens[val_mask], all_labels[val_mask]
    X_test, y_test = all_tokens[test_mask], all_labels[test_mask]

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Normalize per token position and feature
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    num_tokens = X_train.shape[1]
    print(f"Tokens per recording: {num_tokens}")

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=args.batch_size)

    counts = np.bincount(y_train).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / counts, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = ECGTransformer(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_tokens=num_tokens,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, patience_counter = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_labels_arr, val_preds, val_scores = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels_arr, val_preds, zero_division=0)
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
    val_labels_arr, val_preds, val_scores = evaluate(model, val_loader, device)
    test_labels_arr, test_preds, test_scores = evaluate(model, test_loader, device)

    best_threshold = tune_threshold(val_labels_arr, val_scores)
    print(f"Best threshold: {best_threshold:.2f}")
    tuned_val_pred = (val_scores >= best_threshold).astype(int)
    tuned_test_pred = (test_scores >= best_threshold).astype(int)

    summary = {
        "classifier": "transformer",
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "window_sec": args.window_sec,
        "overlap": args.overlap,
        "leads": args.leads,
        "num_tokens": num_tokens,
        "input_dim": input_dim,
        "total_params": total_params,
        "best_threshold": best_threshold,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "val": compute_metrics(val_labels_arr, val_preds, val_scores),
        "test": compute_metrics(test_labels_arr, test_preds, test_scores),
        "val_tuned": compute_metrics(val_labels_arr, tuned_val_pred, val_scores),
        "test_tuned": compute_metrics(test_labels_arr, tuned_test_pred, test_scores),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nTest metrics (default 0.5):")
    print(f"  f1={summary['test']['f1']:.3f} recall={summary['test']['recall']:.3f} roc_auc={summary['test']['roc_auc']:.3f}")
    print(f"Test metrics (tuned {best_threshold:.2f}):")
    print(f"  f1={summary['test_tuned']['f1']:.3f} recall={summary['test_tuned']['recall']:.3f} roc_auc={summary['test_tuned']['roc_auc']:.3f}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
