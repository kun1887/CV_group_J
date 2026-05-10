"""Train a Transformer on raw PTB-XL ECG signals for multi-label classification
over the five diagnostic super-classes (NORM, MI, STTC, CD, HYP).

This script intentionally mirrors the architecture from the team's Slack/WhatsApp
discussion (raw lead-0 ECG, 1s windows tokenized linearly to 256-d, prepended
[CLS], sinusoidal positional encoding, 4 TransformerEncoder layers with 8 heads
and FFN=1024), and only swaps the binary head for a 5-way multi-label head
trained with BCEWithLogitsLoss + class-balanced pos_weight.

Standard PTB-XL evaluation protocol:
- train: strat_fold 1..8
- val:   strat_fold 9   (per-class threshold tuning)
- test:  strat_fold 10  (final reported metrics)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wfdb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

PTBXL_DB_NAME = "ptb-xl"
PTBXL_METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv")
SUPER_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def ensure_ptbxl_metadata(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in PTBXL_METADATA_FILES if not (data_dir / name).exists()]
    if not missing:
        return
    print(f"Downloading PTB-XL metadata into {data_dir}: {', '.join(missing)}")
    wfdb.dl_files(PTBXL_DB_NAME, dl_dir=str(data_dir), files=list(missing), keep_subdirs=False)


def load_scp_to_super_class(data_dir: Path) -> dict[str, str]:
    path = data_dir / "scp_statements.csv"
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = row[reader.fieldnames[0]].strip()
            super_class = (row.get("diagnostic_class") or "").strip()
            if super_class in SUPER_CLASSES:
                mapping[code] = super_class
    return mapping


def encode_super_class_labels(scp_codes_str: str, scp_to_super: dict[str, str]) -> np.ndarray:
    label = np.zeros(len(SUPER_CLASSES), dtype=np.float32)
    parsed = ast.literal_eval(scp_codes_str)
    for code in parsed:
        super_class = scp_to_super.get(code)
        if super_class is None:
            continue
        label[SUPER_CLASSES.index(super_class)] = 1.0
    return label


def load_ptbxl_rows(data_dir: Path, scp_to_super: dict[str, str]) -> list[dict]:
    metadata_path = data_dir / "ptbxl_database.csv"
    rows: list[dict] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = encode_super_class_labels(row["scp_codes"], scp_to_super)
            if label.sum() == 0:
                continue
            try:
                fold = int(row["strat_fold"])
            except (KeyError, ValueError):
                continue
            rows.append(
                {
                    "ecg_id": row["ecg_id"],
                    "filename_lr": row["filename_lr"],
                    "strat_fold": fold,
                    "label": label,
                }
            )
    return rows


def ensure_records_available(data_dir: Path, rows: list[dict]) -> None:
    to_download: list[str] = []
    for row in rows:
        base = row["filename_lr"]
        if (data_dir / f"{base}.dat").exists() and (data_dir / f"{base}.hea").exists():
            continue
        to_download.extend([f"{base}.dat", f"{base}.hea"])
    if not to_download:
        return
    print(f"Downloading {len(to_download) // 2} PTB-XL records into {data_dir}")
    wfdb.dl_files(PTBXL_DB_NAME, dl_dir=str(data_dir), files=to_download, keep_subdirs=True)


class PTBXLRawDataset(Dataset):
    def __init__(self, rows: list[dict], data_dir: Path, lead: int, num_samples: int) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.lead = lead
        self.num_samples = num_samples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        record = wfdb.rdrecord(str(self.data_dir / row["filename_lr"]))
        signal = record.p_signal[: self.num_samples, self.lead].astype(np.float32)
        if signal.shape[0] < self.num_samples:
            signal = np.pad(signal, (0, self.num_samples - signal.shape[0]))
        return torch.from_numpy(signal), torch.from_numpy(row["label"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ECGTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_samples: int = 1000,
        token_size: int = 100,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        num_classes: int = len(SUPER_CLASSES),
    ) -> None:
        super().__init__()
        if num_samples % token_size != 0:
            raise ValueError(f"num_samples ({num_samples}) must be divisible by token_size ({token_size}).")
        self.num_samples = num_samples
        self.token_size = token_size
        self.num_tokens = num_samples // token_size
        self.proj = nn.Linear(token_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.register_buffer("pos_enc", self._build_sinusoidal_pos_enc(self.num_tokens + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _build_sinusoidal_pos_enc(length: int, dim: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        batch = signal.size(0)
        tokens = signal.view(batch, self.num_tokens, self.token_size)
        tokens = self.proj(tokens)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_enc[:, : tokens.size(1)]
        encoded = self.encoder(tokens)
        cls_out = self.norm(encoded[:, 0])
        return self.head(self.head_dropout(cls_out))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_rows_by_fold(rows: list[dict]) -> dict[str, list[dict]]:
    train = [row for row in rows if 1 <= row["strat_fold"] <= 8]
    val = [row for row in rows if row["strat_fold"] == 9]
    test = [row for row in rows if row["strat_fold"] == 10]
    return {"train": train, "val": val, "test": test}


def compute_pos_weight(rows: list[dict]) -> torch.Tensor:
    labels = np.stack([row["label"] for row in rows], axis=0)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for signal, label in loader:
        signal = signal.to(device)
        logits = model(signal)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(label.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def macro_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    aucs = []
    for class_index in range(labels.shape[1]):
        if len(np.unique(labels[:, class_index])) < 2:
            continue
        aucs.append(roc_auc_score(labels[:, class_index], scores[:, class_index]))
    return float(np.mean(aucs)) if aucs else float("nan")


def tune_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(labels.shape[1], dtype=np.float32)
    candidate_grid = np.linspace(0.05, 0.95, 19)
    for class_index in range(labels.shape[1]):
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in candidate_grid:
            preds = (scores[:, class_index] >= threshold).astype(np.int32)
            current_f1 = f1_score(labels[:, class_index], preds, zero_division=0)
            if current_f1 > best_f1:
                best_f1, best_threshold = current_f1, float(threshold)
        thresholds[class_index] = best_threshold
    return thresholds


def per_class_report(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))
    report = []
    for class_index, class_name in enumerate(SUPER_CLASSES):
        preds = (sigmoid_scores[:, class_index] >= thresholds[class_index]).astype(np.int32)
        true = labels[:, class_index].astype(np.int32)
        single_auc = (
            roc_auc_score(true, sigmoid_scores[:, class_index])
            if len(np.unique(true)) == 2
            else None
        )
        report.append(
            {
                "class": class_name,
                "support": int(true.sum()),
                "threshold": float(thresholds[class_index]),
                "auc": None if single_auc is None else float(single_auc),
                "ap": float(average_precision_score(true, sigmoid_scores[:, class_index])),
                "precision": float(precision_score(true, preds, zero_division=0)),
                "recall": float(recall_score(true, preds, zero_division=0)),
                "f1": float(f1_score(true, preds, zero_division=0)),
                "confusion_matrix": confusion_matrix(true, preds, labels=[0, 1]).tolist(),
            }
        )
    return report


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    device: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    num_workers: int
    seed: int
    lead: int
    num_samples: int
    token_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float
    max_records: int | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("src/data/ptbxl"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/data/ptbxl_transformer_multilabel"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lead", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--token-size", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, help="Optional limit for fast smoke tests.")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Device: {device}")

    ensure_ptbxl_metadata(config.data_dir)
    scp_to_super = load_scp_to_super_class(config.data_dir)
    print(f"Loaded {len(scp_to_super)} SCP codes mapped to diagnostic super-classes.")

    rows = load_ptbxl_rows(config.data_dir, scp_to_super)
    if config.max_records is not None:
        rows = rows[: config.max_records]
    print(f"Records with at least one diagnostic super-class label: {len(rows)}")

    splits = split_rows_by_fold(rows)
    for split_name, split_rows in splits.items():
        print(f"  {split_name}: {len(split_rows)} records")

    ensure_records_available(config.data_dir, rows)

    train_loader = DataLoader(
        PTBXLRawDataset(splits["train"], config.data_dir, config.lead, config.num_samples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        PTBXLRawDataset(splits["val"], config.data_dir, config.lead, config.num_samples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        PTBXLRawDataset(splits["test"], config.data_dir, config.lead, config.num_samples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ECGTransformer(
        num_samples=config.num_samples,
        token_size=config.token_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    pos_weight = compute_pos_weight(splits["train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_macro_auc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss, num_examples = 0.0, 0
        for signal, label in train_loader:
            signal = signal.to(device)
            label = label.to(device)
            logits = model(signal)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * signal.size(0)
            num_examples += signal.size(0)
        scheduler.step()
        train_loss = running_loss / max(num_examples, 1)

        val_logits, val_labels = evaluate(model, val_loader, device)
        val_sigmoid = 1.0 / (1.0 + np.exp(-val_logits))
        val_macro_auc = macro_auc(val_sigmoid, val_labels)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auc": val_macro_auc})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_macro_auc={val_macro_auc:.4f}")

        if val_macro_auc > best_macro_auc:
            best_macro_auc = val_macro_auc
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without producing a model state.")
    model.load_state_dict(best_state)

    val_logits, val_labels = evaluate(model, val_loader, device)
    val_sigmoid = 1.0 / (1.0 + np.exp(-val_logits))
    thresholds = tune_thresholds(val_sigmoid, val_labels)
    print(f"Tuned thresholds (per class): {dict(zip(SUPER_CLASSES, thresholds.tolist()))}")

    test_logits, test_labels = evaluate(model, test_loader, device)
    test_sigmoid = 1.0 / (1.0 + np.exp(-test_logits))
    test_macro_auc = macro_auc(test_sigmoid, test_labels)
    report = per_class_report(test_logits, test_labels, thresholds)
    macro_f1 = float(np.mean([row["f1"] for row in report]))
    print(f"\nTest macro-AUC: {test_macro_auc:.4f} | macro-F1: {macro_f1:.4f}")
    for row in report:
        print(
            f"  {row['class']}: AUC={row['auc']:.4f} AP={row['ap']:.4f} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
            f"support={row['support']} threshold={row['threshold']:.2f}"
        )

    summary = {
        "device": str(device),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(config).items()},
        "num_records": {split: len(split_rows) for split, split_rows in splits.items()},
        "thresholds": dict(zip(SUPER_CLASSES, thresholds.tolist())),
        "test_macro_auc": test_macro_auc,
        "test_macro_f1": macro_f1,
        "per_class": report,
        "best_val_macro_auc": best_macro_auc,
        "history": history,
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(best_state, config.output_dir / "model.pt")
    print(f"\nSaved summary and weights to {config.output_dir}")


if __name__ == "__main__":
    main()
