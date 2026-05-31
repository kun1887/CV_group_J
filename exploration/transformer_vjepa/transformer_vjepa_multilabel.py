"""Transformer classifier on top of V-JEPA clip embeddings for multi-label
classification of the five PTB-XL diagnostic super-classes (NORM, MI, STTC,
CD, HYP).

The clip embedding cache produced by `build_vjepa_clip_embeddings.py` only
carries the binary NORM-vs-not-NORM label, so this module re-derives a 5-d
multi-label vector for every record from `ptbxl_database.csv` (which lists
`scp_codes`) and `scp_statements.csv` (which maps each SCP code to a
diagnostic super-class).

Usage from a notebook:

    from transformer_vjepa.transformer_vjepa_multilabel import (
        ensure_ptbxl_metadata,
        load_scp_to_super_class,
        load_multilabel_lookup,
        load_clip_sequences_multilabel,
        split_sequences_by_strat_fold,
        TransformerVJEPAMultiLabel,
        train_model,
        evaluate_model,
        SUPER_CLASSES,
    )
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from dataclasses import dataclass, field
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from transformer_vjepa.transformer_vjepa_binary import (
    build_padding_mask,
    resolve_device,
    set_seed,
)

SUPER_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
PTBXL_DB_NAME = "ptb-xl"
PTBXL_METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiLabelClipSequence:
    record_name: str
    label: np.ndarray  # shape (5,) binary float32
    clip_embeddings: np.ndarray  # shape (num_clips, embedding_dim)
    strat_fold: int | None = None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def ensure_ptbxl_metadata(data_dir: Path) -> Path:
    """Download the two PTB-XL metadata CSVs (~5 MB total) into `data_dir` if
    they are missing. Returns the path to the data directory."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in PTBXL_METADATA_FILES if not (data_dir / name).exists()]
    if missing:
        print(f"Downloading PTB-XL metadata into {data_dir}: {', '.join(missing)}")
        wfdb.dl_files(PTBXL_DB_NAME, dl_dir=str(data_dir), files=list(missing), keep_subdirs=False)
    return data_dir


def load_scp_to_super_class(scp_statements_path: Path) -> dict[str, str]:
    """Read `scp_statements.csv` and return mapping from each SCP code to its
    diagnostic super-class (NORM/MI/STTC/CD/HYP). Non-diagnostic codes are
    excluded from the mapping."""

    mapping: dict[str, str] = {}
    with Path(scp_statements_path).open(newline="", encoding="utf-8") as handle:
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


def load_multilabel_lookup(
    ptbxl_metadata_path: Path, scp_to_super: dict[str, str]
) -> dict[str, dict]:
    """Read the PTB-XL metadata CSV and return a dict
    `ecg_id -> {label, strat_fold}`. Records without any diagnostic
    super-class are skipped."""

    lookup: dict[str, dict] = {}
    with Path(ptbxl_metadata_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ecg_id = str(row["ecg_id"])
                fold = int(row["strat_fold"])
            except (KeyError, ValueError):
                continue
            label = encode_super_class_labels(row["scp_codes"], scp_to_super)
            if label.sum() == 0:
                continue
            lookup[ecg_id] = {"label": label, "strat_fold": fold}
    return lookup


# ---------------------------------------------------------------------------
# Clip cache loading
# ---------------------------------------------------------------------------


def load_clip_sequences_multilabel(
    clip_cache_dir: Path,
    label_lookup: dict[str, dict],
) -> list[MultiLabelClipSequence]:
    """Read every per-record .npz file in the V-JEPA clip embedding cache and
    return one MultiLabelClipSequence per record, attaching the 5-d
    super-class label vector and strat_fold from `label_lookup`. Records
    without a diagnostic super-class label are dropped."""

    clip_cache_dir = Path(clip_cache_dir)
    if not clip_cache_dir.is_dir():
        raise FileNotFoundError(f"Clip embedding cache directory not found: {clip_cache_dir}")

    sequences: list[MultiLabelClipSequence] = []
    for cache_path in sorted(clip_cache_dir.glob("*.npz")):
        with np.load(cache_path, allow_pickle=True) as data:
            clip_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            clip_record_names = data["clip_record_names"].tolist()
            clip_segment_ids = data["clip_segment_ids"].tolist()
            clip_indices = data["clip_indices"].tolist()

        grouped: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = {}
        for row_index, (record_name, segment_id, clip_index) in enumerate(
            zip(clip_record_names, clip_segment_ids, clip_indices)
        ):
            key = (str(record_name), int(segment_id))
            grouped.setdefault(key, []).append((int(clip_index), clip_embeddings[row_index]))

        for (record_name, _segment_id), entries in grouped.items():
            info = label_lookup.get(record_name)
            if info is None:
                continue
            ordered = [embedding for _, embedding in sorted(entries, key=lambda item: item[0])]
            sequences.append(
                MultiLabelClipSequence(
                    record_name=record_name,
                    label=info["label"].copy(),
                    clip_embeddings=np.stack(ordered, axis=0).astype(np.float32),
                    strat_fold=int(info["strat_fold"]),
                )
            )
    return sequences


def split_sequences_by_strat_fold(
    sequences: list[MultiLabelClipSequence],
) -> dict[str, list[MultiLabelClipSequence]]:
    train, val, test = [], [], []
    for item in sequences:
        fold = item.strat_fold
        if fold is None:
            continue
        if 1 <= fold <= 8:
            train.append(item)
        elif fold == 9:
            val.append(item)
        elif fold == 10:
            test.append(item)
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------


class MultiLabelClipSequenceDataset(Dataset):
    def __init__(self, sequences: list[MultiLabelClipSequence]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        item = self.sequences[index]
        return (
            torch.from_numpy(item.clip_embeddings).float(),
            torch.from_numpy(item.label).float(),
            item.record_name,
            len(item.clip_embeddings),
        )


def collate_multilabel_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    embeddings = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    record_names = [item[2] for item in batch]
    padded = pad_sequence(embeddings, batch_first=True)
    return padded, lengths, labels, record_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransformerVJEPAMultiLabel(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int = 1024,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_clips: int = 32,
        num_classes: int = len(SUPER_CLASSES),
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(embedding_dim, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.register_buffer(
            "pos_enc", self._build_sinusoidal_pos_enc(max_clips + 1, model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(model_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(model_dim, num_classes)

    @staticmethod
    def _build_sinusoidal_pos_enc(length: int, dim: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, clip_embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_clips, _ = clip_embeddings.shape
        tokens = self.input_proj(clip_embeddings)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_enc[:, : tokens.size(1)]
        padding_mask = build_padding_mask(lengths, max_clips, tokens.device)
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        cls_out = self.norm(encoded[:, 0])
        return self.head(self.head_dropout(cls_out))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def compute_pos_weight(sequences: list[MultiLabelClipSequence]) -> torch.Tensor:
    labels = np.stack([item.label for item in sequences], axis=0)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


@dataclass
class MultiLabelTrainingHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_macro_auc: list[float] = field(default_factory=list)
    val_macro_f1_at_05: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_macro_auc: float = -1.0


def _run_inference(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_logits, all_labels, all_records = [], [], []
    with torch.no_grad():
        for padded, lengths, labels, record_names in loader:
            padded = padded.to(device)
            logits = model(padded, lengths)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
            all_records.extend(record_names)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    sigmoid = 1.0 / (1.0 + np.exp(-logits))
    return logits, sigmoid, labels, all_records


def macro_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, list[float | None]]:
    per_class: list[float | None] = []
    for class_index in range(labels.shape[1]):
        column = labels[:, class_index]
        if len(np.unique(column)) < 2:
            per_class.append(None)
            continue
        per_class.append(float(roc_auc_score(column, scores[:, class_index])))
    valid = [value for value in per_class if value is not None]
    return float(np.mean(valid)) if valid else float("nan"), per_class


def macro_f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    f1s = []
    for class_index in range(labels.shape[1]):
        preds = (scores[:, class_index] >= threshold).astype(np.int32)
        f1s.append(f1_score(labels[:, class_index], preds, zero_division=0))
    return float(np.mean(f1s))


def tune_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)
    for class_index in range(labels.shape[1]):
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in grid:
            preds = (scores[:, class_index] >= threshold).astype(np.int32)
            score = f1_score(labels[:, class_index], preds, zero_division=0)
            if score > best_f1:
                best_f1, best_threshold = score, float(threshold)
        thresholds[class_index] = best_threshold
    return thresholds


def per_class_report(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> list[dict]:
    report = []
    for class_index, class_name in enumerate(SUPER_CLASSES):
        true = labels[:, class_index].astype(np.int32)
        preds = (scores[:, class_index] >= thresholds[class_index]).astype(np.int32)
        single_auc = (
            float(roc_auc_score(true, scores[:, class_index]))
            if len(np.unique(true)) == 2
            else None
        )
        report.append(
            {
                "class": class_name,
                "support": int(true.sum()),
                "threshold": float(thresholds[class_index]),
                "auc": single_auc,
                "ap": float(average_precision_score(true, scores[:, class_index])),
                "precision": float(precision_score(true, preds, zero_division=0)),
                "recall": float(recall_score(true, preds, zero_division=0)),
                "f1": float(f1_score(true, preds, zero_division=0)),
                "confusion_matrix": confusion_matrix(true, preds, labels=[0, 1]).tolist(),
            }
        )
    return report


def train_model(
    model: TransformerVJEPAMultiLabel,
    train_sequences: list[MultiLabelClipSequence],
    val_sequences: list[MultiLabelClipSequence],
    *,
    device: torch.device | str = "auto",
    batch_size: int = 32,
    epochs: int = 60,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    grad_clip: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[TransformerVJEPAMultiLabel, MultiLabelTrainingHistory, dict]:
    if isinstance(device, str):
        device = resolve_device(device)
    set_seed(seed)
    model = model.to(device)

    train_loader = DataLoader(
        MultiLabelClipSequenceDataset(train_sequences),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_multilabel_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        MultiLabelClipSequenceDataset(val_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multilabel_batch,
        num_workers=num_workers,
    )

    pos_weight = compute_pos_weight(train_sequences).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = MultiLabelTrainingHistory()
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, total_examples = 0.0, 0
        for padded, lengths, labels, _ in train_loader:
            padded = padded.to(device)
            labels = labels.to(device)
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * padded.size(0)
            total_examples += padded.size(0)
        scheduler.step()
        train_loss = running_loss / max(total_examples, 1)

        val_logits, val_sigmoid, val_labels, _ = _run_inference(model, val_loader, device)
        val_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(val_logits),
                torch.from_numpy(val_labels),
                pos_weight=pos_weight.cpu(),
            ).item()
        )
        val_macro_auc, _ = macro_auc(val_sigmoid, val_labels)
        val_macro_f1 = macro_f1_at_threshold(val_sigmoid, val_labels, 0.5)

        history.epochs.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_macro_auc.append(val_macro_auc)
        history.val_macro_f1_at_05.append(val_macro_f1)

        if verbose:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_macro_auc={val_macro_auc:.4f} "
                f"val_macro_f1@0.5={val_macro_f1:.4f}"
            )

        if val_macro_auc > history.best_val_macro_auc:
            history.best_val_macro_auc = val_macro_auc
            history.best_epoch = epoch
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without recording a best checkpoint.")
    model.load_state_dict(best_state)
    return model, history, best_state


def evaluate_model(
    model: TransformerVJEPAMultiLabel,
    val_sequences: list[MultiLabelClipSequence],
    test_sequences: list[MultiLabelClipSequence],
    *,
    device: torch.device | str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """Tune per-class thresholds on val, then report metrics on test."""

    if isinstance(device, str):
        device = resolve_device(device)
    model = model.to(device)

    val_loader = DataLoader(
        MultiLabelClipSequenceDataset(val_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multilabel_batch,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        MultiLabelClipSequenceDataset(test_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multilabel_batch,
        num_workers=num_workers,
    )

    _, val_sigmoid, val_labels, _ = _run_inference(model, val_loader, device)
    thresholds = tune_thresholds(val_sigmoid, val_labels)

    _, test_sigmoid, test_labels, test_records = _run_inference(model, test_loader, device)
    test_macro_auc, test_per_class_auc = macro_auc(test_sigmoid, test_labels)
    test_report = per_class_report(test_sigmoid, test_labels, thresholds)
    test_macro_f1 = float(np.mean([row["f1"] for row in test_report]))

    return {
        "thresholds": dict(zip(SUPER_CLASSES, thresholds.tolist())),
        "test_macro_auc": test_macro_auc,
        "test_macro_f1": test_macro_f1,
        "test_per_class_auc": dict(zip(SUPER_CLASSES, test_per_class_auc)),
        "test_per_class_report": test_report,
        "test_sigmoid_scores": test_sigmoid,
        "test_true_labels": test_labels,
        "test_record_names": test_records,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-cache-dir",
        type=Path,
        default=Path("src/data/ptbxl_vjepa_clip_embedding_experiments/records"),
    )
    parser.add_argument("--ptbxl-data-dir", type=Path, default=Path("src/data/ptbxl"))
    parser.add_argument("--output-dir", type=Path, default=Path("src/data/transformer_vjepa_multilabel"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-clips", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_ptbxl_metadata(args.ptbxl_data_dir)
    scp_to_super = load_scp_to_super_class(args.ptbxl_data_dir / "scp_statements.csv")
    label_lookup = load_multilabel_lookup(args.ptbxl_data_dir / "ptbxl_database.csv", scp_to_super)
    sequences = load_clip_sequences_multilabel(args.clip_cache_dir, label_lookup)
    splits = split_sequences_by_strat_fold(sequences)

    for split_name, items in splits.items():
        if not items:
            continue
        labels = np.stack([item.label for item in items], axis=0)
        counts = dict(zip(SUPER_CLASSES, labels.sum(axis=0).astype(int).tolist()))
        print(f"  {split_name}: {len(items)} sequences | per-class positives={counts}")

    embedding_dim = int(splits["train"][0].clip_embeddings.shape[1])
    model = TransformerVJEPAMultiLabel(
        embedding_dim=embedding_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_clips=args.max_clips,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model, history, best_state = train_model(
        model,
        splits["train"],
        splits["val"],
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    results = evaluate_model(
        model,
        splits["val"],
        splits["test"],
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\nTest macro-AUC: {results['test_macro_auc']:.4f} | macro-F1: {results['test_macro_f1']:.4f}")
    for row in results["test_per_class_report"]:
        auc_text = f"{row['auc']:.4f}" if row["auc"] is not None else "n/a"
        print(
            f"  {row['class']}: AUC={auc_text} AP={row['ap']:.4f} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
            f"support={row['support']} threshold={row['threshold']:.2f}"
        )

    summary = {
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "num_sequences": {split: len(items) for split, items in splits.items()},
        "thresholds": results["thresholds"],
        "test_macro_auc": results["test_macro_auc"],
        "test_macro_f1": results["test_macro_f1"],
        "test_per_class_auc": results["test_per_class_auc"],
        "test_per_class_report": results["test_per_class_report"],
        "best_epoch": history.best_epoch,
        "best_val_macro_auc": history.best_val_macro_auc,
        "history": {
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_macro_auc": history.val_macro_auc,
            "val_macro_f1_at_05": history.val_macro_f1_at_05,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        args.output_dir / "test_predictions.npz",
        record_names=np.asarray(results["test_record_names"]),
        sigmoid_scores=results["test_sigmoid_scores"],
        true_labels=results["test_true_labels"],
        super_classes=np.asarray(SUPER_CLASSES),
    )
    torch.save(best_state, args.output_dir / "model.pt")
    print(f"\nSaved summary and weights to {args.output_dir}")


if __name__ == "__main__":
    main()
