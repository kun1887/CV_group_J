from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from vjepa_embedding_utils import ensure_clip_embedding_cache, resolve_device


@dataclass(frozen=True)
class SegmentSequence:
    record_name: str
    segment_id: int
    label_id: int
    label_text: str
    rhythm: str
    clip_embeddings: np.ndarray
    strat_fold: int | None = None
    sequence_id: str | None = None


class SegmentSequenceDataset(Dataset):
    def __init__(self, sequences: list[SegmentSequence]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str, int]:
        item = self.sequences[index]
        return (
            torch.from_numpy(item.clip_embeddings).float(),
            item.label_id,
            item.segment_id,
            item.sequence_id or item.record_name,
            len(item.clip_embeddings),
        )


class TinyLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny LSTM on V-JEPA clip embeddings and evaluate on one held-out record."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/data/mitdb_vjepa_frames"),
        help="Root folder containing per-record segment exports.",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("src/data/vjepa_clip_embedding_experiments/records"),
        help="Clip embedding cache directory. One .npz file per record is expected unless a legacy .npz path is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data/vjepa_tiny_lstm"),
        help="Where to save metrics and predictions.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/vjepa2-vitl-fpc64-256",
        help="Hugging Face V-JEPA model name used if clip embeddings must be recomputed.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used for LSTM training and any embedding recomputation.",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Optional subset of exported records to use.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Optional limit on the number of segments to use.",
    )
    parser.add_argument(
        "--held-out-record",
        help="Optional held-out record to evaluate on. If omitted, the most balanced eligible record is chosen automatically.",
    )
    parser.add_argument(
        "--min-clips-per-segment",
        type=int,
        default=1,
        help="Minimum number of clip embeddings required for a segment to be kept. Default keeps all segments.",
    )
    parser.add_argument(
        "--train-subsequence-length",
        type=int,
        help=(
            "Optional maximum number of clip embeddings per training sample. "
            "If set, training segments longer than this are split into smaller subsequences."
        ),
    )
    parser.add_argument(
        "--train-subsequence-stride",
        type=int,
        help=(
            "Stride between training subsequences in clip-embedding units. "
            "Defaults to the subsequence length when --train-subsequence-length is set."
        ),
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
        "--embedding-batch-size",
        type=int,
        default=2,
        help="Number of clips per V-JEPA forward pass if clip embeddings must be recomputed.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Number of segment sequences per LSTM training batch.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="LSTM hidden size.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of LSTM layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="LSTM dropout. Ignored when num-layers=1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.25,
        help="Fraction of training segments reserved for validation when stratification is possible.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached clip embeddings and recompute them.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_segment_metadata_fields(dataset_root: Path) -> dict[tuple[str, int], dict[str, object]]:
    metadata_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for metadata_path in sorted(dataset_root.glob("*/segment_*/metadata.json")):
        payload = json.loads(metadata_path.read_text())
        key = (str(payload["record_name"]), int(payload["segment_id"]))
        metadata_by_key[key] = payload
    if not metadata_by_key:
        raise ValueError(f"No segment metadata found under {dataset_root}.")
    return metadata_by_key


def build_segment_sequences(
    clip_embeddings: np.ndarray,
    meta: dict[str, np.ndarray],
    dataset_root: Path,
) -> list[SegmentSequence]:
    grouped: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = {}
    clip_records = meta["clip_record_names"].tolist()
    clip_segment_ids = meta["clip_segment_ids"].tolist()
    clip_labels = meta["clip_labels"].tolist()
    clip_label_text = meta["clip_label_text"].tolist()
    clip_rhythms = meta["clip_rhythms"].tolist()
    clip_indices = meta["clip_indices"].tolist()

    sequence_meta: dict[tuple[str, int], tuple[int, str, str]] = {}
    for row_index, (record_name, segment_id, label_id, label_text, rhythm, clip_index) in enumerate(
        zip(clip_records, clip_segment_ids, clip_labels, clip_label_text, clip_rhythms, clip_indices)
    ):
        key = (str(record_name), int(segment_id))
        grouped.setdefault(key, []).append((int(clip_index), clip_embeddings[row_index]))
        sequence_meta[key] = (int(label_id), str(label_text), str(rhythm))

    metadata_by_key = load_segment_metadata_fields(dataset_root)
    sequences: list[SegmentSequence] = []
    for record_name, segment_id in sorted(grouped):
        ordered_embeddings = [embedding for _, embedding in sorted(grouped[(record_name, segment_id)], key=lambda item: item[0])]
        label_id, label_text, rhythm = sequence_meta[(record_name, segment_id)]
        metadata_payload = metadata_by_key.get((record_name, segment_id))
        if metadata_payload is None:
            raise ValueError(
                f"Missing metadata.json for record {record_name} segment {segment_id} under {dataset_root}."
            )
        strat_fold = metadata_payload.get("strat_fold")
        sequences.append(
            SegmentSequence(
                record_name=record_name,
                segment_id=segment_id,
                label_id=label_id,
                label_text=label_text,
                rhythm=rhythm,
                clip_embeddings=np.stack(ordered_embeddings, axis=0).astype(np.float32),
                strat_fold=None if strat_fold is None else int(strat_fold),
            )
        )
    return sequences


def choose_balanced_held_out_record(sequences: list[SegmentSequence]) -> str:
    counts: dict[str, np.ndarray] = {}
    for item in sequences:
        counts.setdefault(item.record_name, np.zeros(2, dtype=np.int32))[item.label_id] += 1

    candidates: list[tuple[float, int, str]] = []
    for record_name, label_counts in counts.items():
        if label_counts.min() == 0:
            continue
        total = int(label_counts.sum())
        imbalance = float(abs(int(label_counts[0]) - int(label_counts[1])) / total)
        candidates.append((imbalance, -total, record_name))

    if not candidates:
        raise ValueError("No held-out record contains both classes in the available clip-embedding dataset.")

    candidates.sort()
    return candidates[0][2]


def normalize_sequences(
    train_sequences: list[SegmentSequence],
    other_sequences: list[SegmentSequence],
) -> tuple[list[SegmentSequence], list[SegmentSequence], np.ndarray, np.ndarray]:
    train_matrix = np.concatenate([item.clip_embeddings for item in train_sequences], axis=0)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-6] = 1.0

    def apply(items: list[SegmentSequence]) -> list[SegmentSequence]:
        normalized: list[SegmentSequence] = []
        for item in items:
            normalized.append(
                SegmentSequence(
                    record_name=item.record_name,
                    segment_id=item.segment_id,
                    label_id=item.label_id,
                    label_text=item.label_text,
                    rhythm=item.rhythm,
                    clip_embeddings=((item.clip_embeddings - mean) / std).astype(np.float32),
                    strat_fold=item.strat_fold,
                    sequence_id=item.sequence_id,
                )
            )
        return normalized

    return apply(train_sequences), apply(other_sequences), mean, std


def split_train_val(
    train_sequences: list[SegmentSequence],
    val_fraction: float,
    seed: int,
) -> tuple[list[SegmentSequence], list[SegmentSequence]]:
    labels = np.array([item.label_id for item in train_sequences], dtype=np.int32)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or counts.min() < 2 or len(train_sequences) < 6:
        return train_sequences, []

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(train_sequences)), labels))
    train_items = [train_sequences[idx] for idx in train_idx]
    val_items = [train_sequences[idx] for idx in val_idx]
    return train_items, val_items


def split_long_training_sequences(
    train_sequences: list[SegmentSequence],
    subsequence_length: int | None,
    subsequence_stride: int | None,
) -> tuple[list[SegmentSequence], dict[str, int]]:
    if subsequence_length is None:
        return train_sequences, {
            "raw_train_segments": len(train_sequences),
            "effective_train_samples": len(train_sequences),
            "segments_split": 0,
            "generated_subsequences": 0,
            "subsequence_length": 0,
            "subsequence_stride": 0,
        }

    if subsequence_length < 1:
        raise ValueError("--train-subsequence-length must be at least 1.")
    stride = subsequence_stride if subsequence_stride is not None else subsequence_length
    if stride < 1:
        raise ValueError("--train-subsequence-stride must be at least 1.")

    expanded: list[SegmentSequence] = []
    segments_split = 0
    generated_subsequences = 0

    for item in train_sequences:
        length = len(item.clip_embeddings)
        if length <= subsequence_length:
            expanded.append(item)
            continue

        segments_split += 1
        start_indices = list(range(0, length - subsequence_length + 1, stride))
        last_start = length - subsequence_length
        if start_indices[-1] != last_start:
            start_indices.append(last_start)

        for subseq_index, start_idx in enumerate(start_indices):
            end_idx = start_idx + subsequence_length
            expanded.append(
                SegmentSequence(
                    record_name=item.record_name,
                    segment_id=item.segment_id,
                    label_id=item.label_id,
                    label_text=item.label_text,
                    rhythm=item.rhythm,
                    clip_embeddings=item.clip_embeddings[start_idx:end_idx].astype(np.float32),
                    strat_fold=item.strat_fold,
                    sequence_id=f"{item.record_name}:{item.segment_id}:subseq_{subseq_index:03d}",
                )
            )
            generated_subsequences += 1

    return expanded, {
        "raw_train_segments": len(train_sequences),
        "effective_train_samples": len(expanded),
        "segments_split": segments_split,
        "generated_subsequences": generated_subsequences,
        "subsequence_length": subsequence_length,
        "subsequence_stride": stride,
    }


def collate_batch(batch: list[tuple[torch.Tensor, int, int, str, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[str]]:
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    segment_ids = [item[2] for item in batch]
    record_names = [item[3] for item in batch]
    lengths = torch.tensor([item[4] for item in batch], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, labels, segment_ids, record_names


def evaluate_model(
    model: TinyLSTMClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_scores: list[float] = []
    all_segment_ids: list[int] = []
    all_record_names: list[str] = []
    total_loss = 0.0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for padded, lengths, labels, segment_ids, record_names in loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_scores.extend(probabilities.cpu().tolist())
            all_segment_ids.extend(segment_ids)
            all_record_names.extend(record_names)

    results: dict[str, object] = {
        "loss": total_loss / max(total_examples, 1),
        "labels": np.array(all_labels, dtype=np.int32),
        "predictions": np.array(all_predictions, dtype=np.int32),
        "scores": np.array(all_scores, dtype=np.float32),
        "segment_ids": all_segment_ids,
        "record_names": all_record_names,
    }
    if total_examples:
        labels_arr = results["labels"]
        preds_arr = results["predictions"]
        scores_arr = results["scores"]
        results.update(
            {
                "accuracy": float(accuracy_score(labels_arr, preds_arr)),
                "precision": float(precision_score(labels_arr, preds_arr, zero_division=0)),
                "recall": float(recall_score(labels_arr, preds_arr, zero_division=0)),
                "f1": float(f1_score(labels_arr, preds_arr, zero_division=0)),
                "roc_auc": float(roc_auc_score(labels_arr, scores_arr)) if len(np.unique(labels_arr)) == 2 else None,
                "confusion_matrix": confusion_matrix(labels_arr, preds_arr).tolist(),
            }
        )
    return results


def save_prediction_rows(path: Path, eval_results: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = eval_results["labels"]
    predictions = eval_results["predictions"]
    scores = eval_results["scores"]
    segment_ids = eval_results["segment_ids"]
    record_names = eval_results["record_names"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_name", "segment_id", "true_label", "predicted_label", "score_not_normal"],
        )
        writer.writeheader()
        for record_name, segment_id, label, prediction, score in zip(
            record_names, segment_ids, labels.tolist(), predictions.tolist(), scores.tolist()
        ):
            writer.writerow(
                {
                    "record_name": record_name,
                    "segment_id": segment_id,
                    "true_label": int(label),
                    "predicted_label": int(prediction),
                    "score_not_normal": float(score),
                }
            )


def resolve_split_strategy(sequences: list[SegmentSequence]) -> str:
    strat_fold_presence = [item.strat_fold is not None for item in sequences]
    if all(strat_fold_presence):
        return "ptbxl_strat_fold"
    if any(strat_fold_presence):
        raise ValueError(
            "Inconsistent segment metadata: some selected sequences have strat_fold and others do not. "
            "Refuse to mix PTB-XL and non-PTB-XL split strategies."
        )
    return "held_out_record"


def split_by_ptbxl_strat_fold(
    sequences: list[SegmentSequence],
) -> tuple[list[SegmentSequence], list[SegmentSequence], list[SegmentSequence], dict[str, list[int]]]:
    train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
    validation_folds = [9]
    test_folds = [10]

    train_sequences = [item for item in sequences if item.strat_fold in train_folds]
    val_sequences = [item for item in sequences if item.strat_fold in validation_folds]
    test_sequences = [item for item in sequences if item.strat_fold in test_folds]

    if not train_sequences:
        raise ValueError("No PTB-XL training sequences remain after applying canonical strat_fold split.")
    if not val_sequences:
        raise ValueError("No PTB-XL validation sequences remain after applying canonical strat_fold split.")
    if not test_sequences:
        raise ValueError("No PTB-XL test sequences remain after applying canonical strat_fold split.")

    for split_name, split_sequences in [
        ("training", train_sequences),
        ("validation", val_sequences),
        ("test", test_sequences),
    ]:
        if len({item.label_id for item in split_sequences}) < 2:
            raise ValueError(
                f"PTB-XL {split_name} split does not contain both classes after filtering. "
                f"Adjust the selected records or clip-count filter."
            )

    return train_sequences, val_sequences, test_sequences, {
        "train_folds": train_folds,
        "validation_folds": validation_folds,
        "test_folds": test_folds,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    print("Phase 1/5: loading or building clip embeddings.")
    selected_records = set(args.records) if args.records else None
    _, clip_embeddings, clip_meta = ensure_clip_embedding_cache(
        dataset_root=args.dataset_root,
        embedding_cache=args.embedding_cache,
        selected_records=selected_records,
        max_segments=args.max_segments,
        model_name=args.model_name,
        device_arg=args.device,
        batch_size=args.embedding_batch_size,
        target_num_frames=args.target_num_frames,
        clip_stride=args.clip_stride,
        force_recompute=args.force_recompute,
    )

    print("Phase 2/5: building segment-level clip sequences.")
    sequences = build_segment_sequences(clip_embeddings, clip_meta, args.dataset_root)
    if args.min_clips_per_segment < 1:
        raise ValueError("--min-clips-per-segment must be at least 1.")
    if args.min_clips_per_segment > 1:
        original_count = len(sequences)
        sequences = [item for item in sequences if len(item.clip_embeddings) >= args.min_clips_per_segment]
        print(
            f"Filtered segments by min clip count: kept {len(sequences)} / {original_count} "
            f"with at least {args.min_clips_per_segment} clip(s)."
        )
        if not sequences:
            raise ValueError("No segment sequences remain after applying --min-clips-per-segment.")
    if len({item.label_id for item in sequences}) < 2:
        raise ValueError("Training requires both normal and not-normal segment labels.")

    split_strategy = resolve_split_strategy(sequences)
    held_out_record: str | None = None
    split_info: dict[str, list[int] | None] = {
        "train_folds": None,
        "validation_folds": None,
        "test_folds": None,
    }

    if split_strategy == "ptbxl_strat_fold":
        print("Split strategy: ptbxl_strat_fold")
        if args.held_out_record:
            print("Note: --held-out-record is ignored for PTB-XL. Using canonical strat_fold split instead.")
        print("Note: --val-fraction is ignored for PTB-XL. Validation uses strat_fold 9.")
        raw_train_sequences, val_sequences, test_sequences, fold_info = split_by_ptbxl_strat_fold(sequences)
        split_info.update(fold_info)
        print(
            f"Train folds: {','.join(str(fold) for fold in fold_info['train_folds'])} | "
            f"Validation fold: {fold_info['validation_folds'][0]} | "
            f"Test fold: {fold_info['test_folds'][0]}"
        )
    else:
        held_out_record = args.held_out_record or choose_balanced_held_out_record(sequences)
        test_sequences = [item for item in sequences if item.record_name == held_out_record]
        train_sequences = [item for item in sequences if item.record_name != held_out_record]
        if not test_sequences:
            raise ValueError(f"No sequences found for held-out record {held_out_record}.")
        if len({item.label_id for item in test_sequences}) < 2:
            raise ValueError(f"Held-out record {held_out_record} does not contain both classes.")
        if len({item.label_id for item in train_sequences}) < 2:
            raise ValueError("Training split does not contain both classes after applying the held-out record.")
        raw_train_sequences, val_sequences = split_train_val(train_sequences, args.val_fraction, args.seed)

    train_sequences, subsequence_summary = split_long_training_sequences(
        raw_train_sequences,
        args.train_subsequence_length,
        args.train_subsequence_stride,
    )
    if args.train_subsequence_length is not None:
        print(
            f"Expanded long training segments into subsequences: "
            f"{subsequence_summary['effective_train_samples']} training samples from "
            f"{subsequence_summary['raw_train_segments']} train segment(s), "
            f"{subsequence_summary['segments_split']} segment(s) split."
        )
    train_sequences, normalized_other, mean, std = normalize_sequences(train_sequences, val_sequences + test_sequences)
    normalized_val = normalized_other[: len(val_sequences)]
    normalized_test = normalized_other[len(val_sequences) :]

    if split_strategy == "ptbxl_strat_fold":
        print(
            f"train samples={len(train_sequences)} val segments={len(normalized_val)} "
            f"test segments={len(normalized_test)}"
        )
    else:
        print(
            f"Held-out record: {held_out_record} | "
            f"train samples={len(train_sequences)} val segments={len(normalized_val)} test segments={len(normalized_test)}"
        )

    print("Phase 3/5: preparing loaders and tiny LSTM.")
    device = torch.device(resolve_device(args.device))
    train_loader = DataLoader(
        SegmentSequenceDataset(train_sequences),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(
            SegmentSequenceDataset(normalized_val),
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        if normalized_val
        else None
    )
    test_loader = DataLoader(
        SegmentSequenceDataset(normalized_test),
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    input_dim = int(train_sequences[0].clip_embeddings.shape[1])
    model = TinyLSTMClassifier(
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_labels = np.array([item.label_id for item in train_sequences], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1.0), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Phase 4/5: training.")
    best_state = None
    best_score = -np.inf
    patience_counter = 0
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for padded, lengths, labels, _, _ in train_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

        train_loss = total_loss / max(total_examples, 1)
        if val_loader is not None:
            val_results = evaluate_model(model, val_loader, device)
            monitor_value = float(val_results["f1"])
            val_loss = float(val_results["loss"])
        else:
            val_results = None
            monitor_value = -train_loss
            val_loss = None

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": None if val_results is None else float(val_results["f1"]),
                "val_accuracy": None if val_results is None else float(val_results["accuracy"]),
                "val_precision": None if val_results is None else float(val_results["precision"]),
                "val_recall": None if val_results is None else float(val_results["recall"]),
                "val_roc_auc": None if val_results is None else val_results["roc_auc"],
            }
        )
        val_loss_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
        val_f1_text = "n/a" if val_results is None else f"{monitor_value:.4f}"
        val_score_preview = ""
        if val_results is not None and len(val_results["scores"]) > 0:
            preview_scores = ", ".join(f"{score:.3f}" for score in val_results["scores"][:5].tolist())
            val_score_preview = f" val_score_not_normal[:5]=[{preview_scores}]"
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss_text} "
            f"val_f1={val_f1_text}"
            f"{val_score_preview}"
        )

        if monitor_value > best_score:
            best_score = monitor_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without capturing a model state.")
    model.load_state_dict(best_state)

    print("Phase 5/5: evaluating on the held-out record and saving outputs.")
    val_results_final = evaluate_model(model, val_loader, device) if val_loader is not None else None
    test_results = evaluate_model(model, test_loader, device)
    if val_results_final is not None:
        save_prediction_rows(args.output_dir / "validation_predictions.csv", val_results_final)
    save_prediction_rows(args.output_dir / "held_out_predictions.csv", test_results)

    summary = {
        "split_strategy": split_strategy,
        "train_folds": split_info["train_folds"],
        "validation_folds": split_info["validation_folds"],
        "test_folds": split_info["test_folds"],
        "held_out_record": held_out_record,
        "num_total_segments": len(sequences),
        "num_train_segments_before_subsequence_split": subsequence_summary["raw_train_segments"],
        "num_train_samples_after_subsequence_split": len(train_sequences),
        "num_val_segments": len(normalized_val),
        "num_test_segments": len(normalized_test),
        "train_label_distribution": {
            "normal_rhythm": int((train_labels == 0).sum()),
            "not_normal_rhythm": int((train_labels == 1).sum()),
        },
        "test_label_distribution": {
            "normal_rhythm": int((test_results["labels"] == 0).sum()),
            "not_normal_rhythm": int((test_results["labels"] == 1).sum()),
        },
        "input_dim": input_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "min_clips_per_segment": args.min_clips_per_segment,
        "train_subsequence_length": args.train_subsequence_length,
        "train_subsequence_stride": subsequence_summary["subsequence_stride"],
        "segments_split_for_training": subsequence_summary["segments_split"],
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "metrics": {
            "accuracy": float(test_results["accuracy"]),
            "precision": float(test_results["precision"]),
            "recall": float(test_results["recall"]),
            "f1": float(test_results["f1"]),
            "roc_auc": None if test_results["roc_auc"] is None else float(test_results["roc_auc"]),
        },
        "validation_metrics": None
        if val_results_final is None
        else {
            "accuracy": float(val_results_final["accuracy"]),
            "precision": float(val_results_final["precision"]),
            "recall": float(val_results_final["recall"]),
            "f1": float(val_results_final["f1"]),
            "roc_auc": None if val_results_final["roc_auc"] is None else float(val_results_final["roc_auc"]),
        },
        "confusion_matrix": test_results["confusion_matrix"],
        "normalization": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
        "history": history,
    }
    (args.output_dir / "tiny_lstm_summary.json").write_text(json.dumps(summary, indent=2))

    if split_strategy == "ptbxl_strat_fold":
        print(
            "Evaluation split: "
            f"train folds {split_info['train_folds']}, "
            f"validation folds {split_info['validation_folds']}, "
            f"test folds {split_info['test_folds']}"
        )
    else:
        print(f"Held-out record: {held_out_record}")
    print(
        "Test metrics: "
        f"accuracy={test_results['accuracy']:.3f} "
        f"precision={test_results['precision']:.3f} "
        f"recall={test_results['recall']:.3f} "
        f"f1={test_results['f1']:.3f} "
        f"roc_auc={test_results['roc_auc'] if test_results['roc_auc'] is not None else 'n/a'}"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
