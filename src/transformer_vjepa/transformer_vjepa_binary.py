"""Transformer classifier on top of V-JEPA clip embeddings for binary
normal-vs-not-normal rhythm classification on PTB-XL.

The pipeline reuses the clip embedding cache already produced by
`build_vjepa_clip_embeddings.py` on the PTB-XL frame export. Each record
contributes a variable-length sequence of clip embeddings (1024-d), labeled
NORM (0) or not-NORM (1) at the record level.

This module is intentionally importable from a notebook. The standard usage
pattern is:

    from transformer_vjepa.transformer_vjepa_binary import (
        load_clip_sequences,
        split_sequences_by_strat_fold,
        TransformerVJEPABinary,
        train_model,
        evaluate_model,
    )

    sequences = load_clip_sequences(...)
    splits = split_sequences_by_strat_fold(sequences, ptbxl_metadata)
    model = TransformerVJEPABinary(...)
    history = train_model(model, splits["train"], splits["val"], ...)
    test_metrics, threshold = evaluate_model(model, splits["val"], splits["test"], ...)

The same module also exposes a CLI entry point so it can be run end-to-end
from the command line for headless training runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import wfdb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipSequence:
    record_name: str
    label_id: int
    label_text: str
    clip_embeddings: np.ndarray  # shape: (num_clips, embedding_dim)
    strat_fold: int | None = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_clip_sequences(
    clip_cache_dir: Path,
    *,
    selected_records: set[str] | None = None,
) -> list[ClipSequence]:
    """Read every per-record .npz file in the V-JEPA clip embedding cache and
    return one ClipSequence per record."""

    clip_cache_dir = Path(clip_cache_dir)
    if not clip_cache_dir.is_dir():
        raise FileNotFoundError(f"Clip embedding cache directory not found: {clip_cache_dir}")

    sequences: list[ClipSequence] = []
    for cache_path in sorted(clip_cache_dir.glob("*.npz")):
        with np.load(cache_path, allow_pickle=True) as data:
            clip_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            clip_record_names = data["clip_record_names"].tolist()
            clip_segment_ids = data["clip_segment_ids"].tolist()
            clip_labels = data["clip_labels"].tolist()
            clip_label_text = data["clip_label_text"].tolist()
            clip_indices = data["clip_indices"].tolist()

        grouped: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = {}
        sequence_meta: dict[tuple[str, int], tuple[int, str]] = {}
        for row_index, (record_name, segment_id, label_id, label_text, clip_index) in enumerate(
            zip(clip_record_names, clip_segment_ids, clip_labels, clip_label_text, clip_indices)
        ):
            key = (str(record_name), int(segment_id))
            grouped.setdefault(key, []).append((int(clip_index), clip_embeddings[row_index]))
            sequence_meta[key] = (int(label_id), str(label_text))

        for (record_name, _segment_id), entries in grouped.items():
            if selected_records is not None and record_name not in selected_records:
                continue
            ordered = [embedding for _, embedding in sorted(entries, key=lambda item: item[0])]
            label_id, label_text = sequence_meta[(record_name, _segment_id)]
            sequences.append(
                ClipSequence(
                    record_name=record_name,
                    label_id=label_id,
                    label_text=label_text,
                    clip_embeddings=np.stack(ordered, axis=0).astype(np.float32),
                )
            )

    return sequences


def ensure_ptbxl_metadata(metadata_path: Path) -> Path:
    """Make sure `ptbxl_database.csv` exists at the requested location and
    download it from PhysioNet if missing. The metadata CSV is small (~5 MB)
    and is the only PTB-XL file needed when the clip embedding cache already
    exists."""

    metadata_path = Path(metadata_path)
    if metadata_path.exists():
        return metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ptbxl_database.csv into {metadata_path.parent}")
    wfdb.dl_files(
        "ptb-xl",
        dl_dir=str(metadata_path.parent),
        files=["ptbxl_database.csv"],
        keep_subdirs=False,
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Failed to download PTB-XL metadata to {metadata_path}")
    return metadata_path


def load_strat_fold_lookup(ptbxl_metadata_path: Path) -> dict[str, int]:
    """Build an ecg_id -> strat_fold mapping from the PTB-XL metadata CSV."""

    lookup: dict[str, int] = {}
    with Path(ptbxl_metadata_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                lookup[str(row["ecg_id"])] = int(row["strat_fold"])
            except (KeyError, ValueError):
                continue
    return lookup


def attach_strat_fold(
    sequences: list[ClipSequence], fold_lookup: dict[str, int]
) -> list[ClipSequence]:
    enriched: list[ClipSequence] = []
    for item in sequences:
        fold = fold_lookup.get(item.record_name)
        enriched.append(
            ClipSequence(
                record_name=item.record_name,
                label_id=item.label_id,
                label_text=item.label_text,
                clip_embeddings=item.clip_embeddings,
                strat_fold=fold,
            )
        )
    return enriched


def split_sequences_by_strat_fold(
    sequences: list[ClipSequence],
) -> dict[str, list[ClipSequence]]:
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


class ClipSequenceDataset(Dataset):
    def __init__(self, sequences: list[ClipSequence]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, int]:
        item = self.sequences[index]
        return (
            torch.from_numpy(item.clip_embeddings).float(),
            item.label_id,
            item.record_name,
            len(item.clip_embeddings),
        )


def collate_clip_batch(
    batch: list[tuple[torch.Tensor, int, str, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    embeddings = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    record_names = [item[2] for item in batch]
    padded = pad_sequence(embeddings, batch_first=True)
    return padded, lengths, labels, record_names


def build_padding_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of shape (batch, max_length + 1) where True marks padded
    positions. The +1 accounts for the prepended CLS token (never padded)."""

    batch_size = lengths.size(0)
    range_row = torch.arange(max_length, device=device).unsqueeze(0)
    padding = range_row >= lengths.unsqueeze(1).to(device)
    cls_column = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
    return torch.cat([cls_column, padding], dim=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TransformerVJEPABinary(nn.Module):
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
        num_classes: int = 2,
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
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

    def forward(
        self,
        clip_embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
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
# Training and evaluation
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


def compute_class_weights(sequences: list[ClipSequence], num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for item in sequences:
        counts[item.label_id] += 1.0
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (counts * num_classes)
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class TrainingHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_roc_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = -1.0


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
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
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits, probabilities, labels, all_records


def _compute_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    positive_scores = probabilities[:, 1]
    predictions = (positive_scores >= threshold).astype(np.int32)
    roc_auc = float(roc_auc_score(labels, positive_scores)) if len(np.unique(labels)) == 2 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "num_examples": int(len(labels)),
        "positive_rate_true": float(labels.mean()),
        "positive_rate_pred": float(predictions.mean()),
    }


def tune_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 19)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        predictions = (probabilities[:, 1] >= threshold).astype(np.int32)
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_threshold


def train_model(
    model: TransformerVJEPABinary,
    train_sequences: list[ClipSequence],
    val_sequences: list[ClipSequence],
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
) -> tuple[TransformerVJEPABinary, TrainingHistory, dict]:
    """Train the model and return the best-checkpointed model, the training
    history, and the state dict of the best checkpoint."""

    if isinstance(device, str):
        device = resolve_device(device)
    set_seed(seed)
    model = model.to(device)

    train_loader = DataLoader(
        ClipSequenceDataset(train_sequences),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_clip_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        ClipSequenceDataset(val_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_clip_batch,
        num_workers=num_workers,
    )

    class_weights = compute_class_weights(train_sequences).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = TrainingHistory()
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

        val_logits, val_probs, val_labels, _ = _run_inference(model, val_loader, device)
        val_loss = float(
            nn.functional.cross_entropy(
                torch.from_numpy(val_logits),
                torch.from_numpy(val_labels).long(),
                weight=class_weights.cpu(),
            ).item()
        )
        val_metrics = _compute_metrics(val_probs, val_labels, threshold=0.5)

        history.epochs.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_metrics["accuracy"])
        history.val_f1.append(val_metrics["f1"])
        history.val_roc_auc.append(val_metrics["roc_auc"] or float("nan"))

        if verbose:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_auc={(val_metrics['roc_auc'] or float('nan')):.4f}"
            )

        if val_metrics["f1"] > history.best_val_f1:
            history.best_val_f1 = val_metrics["f1"]
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
    model: TransformerVJEPABinary,
    val_sequences: list[ClipSequence],
    test_sequences: list[ClipSequence],
    *,
    device: torch.device | str = "auto",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[dict, float, dict]:
    """Tune threshold on val, then report metrics on test. Returns
    (test_metrics, tuned_threshold, val_metrics_at_tuned_threshold)."""

    if isinstance(device, str):
        device = resolve_device(device)
    model = model.to(device)

    val_loader = DataLoader(
        ClipSequenceDataset(val_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_clip_batch,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        ClipSequenceDataset(test_sequences),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_clip_batch,
        num_workers=num_workers,
    )

    _, val_probs, val_labels, _ = _run_inference(model, val_loader, device)
    threshold = tune_threshold(val_probs, val_labels)
    val_metrics = _compute_metrics(val_probs, val_labels, threshold=threshold)

    _, test_probs, test_labels, test_records = _run_inference(model, test_loader, device)
    test_metrics = _compute_metrics(test_probs, test_labels, threshold=threshold)
    test_metrics["positive_scores"] = test_probs[:, 1].tolist()
    test_metrics["true_labels"] = test_labels.astype(int).tolist()
    test_metrics["record_names"] = test_records
    return test_metrics, threshold, val_metrics


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
    parser.add_argument(
        "--ptbxl-metadata",
        type=Path,
        default=Path("src/data/ptbxl/ptbxl_database.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("src/data/transformer_vjepa_binary"))
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

    sequences = load_clip_sequences(args.clip_cache_dir)
    fold_lookup = load_strat_fold_lookup(args.ptbxl_metadata)
    sequences = attach_strat_fold(sequences, fold_lookup)
    splits = split_sequences_by_strat_fold(sequences)

    label_counts = {"train": {}, "val": {}, "test": {}}
    for split_name, items in splits.items():
        for item in items:
            label_counts[split_name][item.label_id] = label_counts[split_name].get(item.label_id, 0) + 1
        print(f"  {split_name}: {len(items)} sequences | label counts={label_counts[split_name]}")

    embedding_dim = int(sequences[0].clip_embeddings.shape[1])
    model = TransformerVJEPABinary(
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

    test_metrics, threshold, val_metrics = evaluate_model(
        model,
        splits["val"],
        splits["test"],
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(
        f"\nTuned threshold: {threshold:.2f} | "
        f"val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['accuracy']:.4f} "
        f"val_auc={(val_metrics['roc_auc'] or float('nan')):.4f}"
    )
    print(
        f"Test: f1={test_metrics['f1']:.4f} acc={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
        f"auc={(test_metrics['roc_auc'] or float('nan')):.4f}"
    )
    print(f"Test confusion matrix: {test_metrics['confusion_matrix']}")

    summary = {
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "num_sequences": {split: len(items) for split, items in splits.items()},
        "label_counts": label_counts,
        "embedding_dim": embedding_dim,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": {
            key: value
            for key, value in test_metrics.items()
            if key not in {"positive_scores", "true_labels", "record_names"}
        },
        "best_epoch": history.best_epoch,
        "best_val_f1": history.best_val_f1,
        "history": {
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_accuracy": history.val_accuracy,
            "val_f1": history.val_f1,
            "val_roc_auc": history.val_roc_auc,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        args.output_dir / "test_predictions.npz",
        record_names=np.asarray(test_metrics["record_names"]),
        positive_scores=np.asarray(test_metrics["positive_scores"]),
        true_labels=np.asarray(test_metrics["true_labels"]),
    )
    torch.save(best_state, args.output_dir / "model.pt")
    print(f"\nSaved summary and weights to {args.output_dir}")


if __name__ == "__main__":
    main()
