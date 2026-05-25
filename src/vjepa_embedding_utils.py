from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor


@dataclass(frozen=True)
class SegmentExample:
    record_name: str
    segment_id: int
    label_text: str
    label_id: int
    rhythm: str
    frames_dir: Path
    num_frames: int


def uses_legacy_single_file_cache(cache_path: Path) -> bool:
    return cache_path.suffix == ".npz"


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def load_segment_examples(
    dataset_root: Path,
    selected_records: set[str] | None,
    max_segments: int | None,
) -> list[SegmentExample]:
    examples: list[SegmentExample] = []
    for metadata_path in sorted(dataset_root.glob("*/segment_*/metadata.json")):
        record_name = metadata_path.parent.parent.name
        if selected_records and record_name not in selected_records:
            continue

        payload = json.loads(metadata_path.read_text())
        frames_dir = dataset_root / payload["frames_dir"]
        label_text = payload["class"]
        label_id = 0 if label_text == "normal rhythm" else 1
        examples.append(
            SegmentExample(
                record_name=record_name,
                segment_id=int(payload["segment_id"]),
                label_text=label_text,
                label_id=label_id,
                rhythm=str(payload["rhythm"]),
                frames_dir=frames_dir,
                num_frames=int(payload["num_frames"]),
            )
        )
        if max_segments is not None and len(examples) >= max_segments:
            break

    if not examples:
        raise ValueError(f"No segment metadata found under {dataset_root}.")
    return examples


def load_segment_frames(frames_dir: Path) -> np.ndarray:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise ValueError(f"No frame PNG files found in {frames_dir}.")

    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to read frame {frame_path}.")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def build_fixed_length_clips(frames: np.ndarray, target_num_frames: int, stride: int) -> list[np.ndarray]:
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("Cannot build clips from an empty frame sequence.")

    if num_frames <= target_num_frames:
        frame_indices = np.linspace(0, num_frames - 1, target_num_frames).round().astype(int)
        return [frames[frame_indices]]

    clips: list[np.ndarray] = []
    start_indices = list(range(0, num_frames - target_num_frames + 1, stride))
    last_start = num_frames - target_num_frames
    if start_indices[-1] != last_start:
        start_indices.append(last_start)

    for start_idx in start_indices:
        end_idx = start_idx + target_num_frames
        clips.append(frames[start_idx:end_idx])
    return clips


def build_fixed_length_clips_with_metadata(
    frames: np.ndarray,
    target_num_frames: int,
    stride: int,
) -> list[tuple[np.ndarray, int, int]]:
    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("Cannot build clips from an empty frame sequence.")

    if num_frames <= target_num_frames:
        frame_indices = np.linspace(0, num_frames - 1, target_num_frames).round().astype(int)
        return [(frames[frame_indices], 0, num_frames)]

    clips: list[tuple[np.ndarray, int, int]] = []
    start_indices = list(range(0, num_frames - target_num_frames + 1, stride))
    last_start = num_frames - target_num_frames
    if start_indices[-1] != last_start:
        start_indices.append(last_start)

    for start_idx in start_indices:
        end_idx = start_idx + target_num_frames
        clips.append((frames[start_idx:end_idx], start_idx, end_idx))
    return clips


def embed_segments(
    examples: list[SegmentExample],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    target_num_frames: int,
    clip_stride: int,
) -> np.ndarray:
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        processor = AutoVideoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    except ImportError as exc:
        raise RuntimeError(
            "V-JEPA video preprocessing requires torchvision. "
            "Install it in the active environment, for example: "
            "`pip install torchvision`, then rerun the script."
        ) from exc
    model.eval()

    segment_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for segment_index, example in enumerate(examples, start=1):
            frames = load_segment_frames(example.frames_dir)
            clips = build_fixed_length_clips(frames, target_num_frames=target_num_frames, stride=clip_stride)
            clip_embeddings: list[np.ndarray] = []

            for batch_start in range(0, len(clips), batch_size):
                batch_clips = clips[batch_start : batch_start + batch_size]
                inputs = processor(batch_clips, return_tensors="pt").to(device)
                outputs = model(**inputs, skip_predictor=True)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                clip_embeddings.append(batch_embeddings)

            pooled = np.concatenate(clip_embeddings, axis=0).mean(axis=0)
            segment_embeddings.append(pooled)

            print(
                f"[{segment_index}/{len(examples)}] "
                f"record={example.record_name} segment={example.segment_id} "
                f"frames={example.num_frames} clips={len(clips)}"
            )

    return np.stack(segment_embeddings, axis=0)


def embed_segment_clips(
    examples: list[SegmentExample],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    target_num_frames: int,
    clip_stride: int,
    on_record_complete: Callable[[str, np.ndarray, dict[str, np.ndarray]], None] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        processor = AutoVideoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    except ImportError as exc:
        raise RuntimeError(
            "V-JEPA video preprocessing requires torchvision. "
            "Install it in the active environment, for example: "
            "`pip install torchvision`, then rerun the script."
        ) from exc
    model.eval()

    clip_embeddings: list[np.ndarray] = []
    meta_chunks: dict[str, list[np.ndarray]] = {}

    record_clip_embeddings: dict[str, list[np.ndarray]] = {}
    record_clip_record_names: dict[str, list[str]] = {}
    record_clip_segment_ids: dict[str, list[int]] = {}
    record_clip_labels: dict[str, list[int]] = {}
    record_clip_label_text: dict[str, list[str]] = {}
    record_clip_rhythms: dict[str, list[str]] = {}
    record_clip_indices: dict[str, list[int]] = {}
    record_clip_start_frames: dict[str, list[int]] = {}
    record_clip_end_frames: dict[str, list[int]] = {}
    record_segment_num_clips: dict[str, list[int]] = {}
    record_segment_record_names: dict[str, list[str]] = {}
    record_segment_ids: dict[str, list[int]] = {}
    record_segment_labels: dict[str, list[int]] = {}
    record_segment_label_text: dict[str, list[str]] = {}
    record_segment_rhythms: dict[str, list[str]] = {}
    record_segment_num_frames: dict[str, list[int]] = {}
    record_expected_clips: dict[str, int] = {}
    record_emitted_clips: dict[str, int] = {}
    closed_records: set[str] = set()
    emitted_records: set[str] = set()

    batch_clips: list[np.ndarray] = []
    batch_meta: list[tuple[str, int, int, str, str, int, int, int]] = []

    def init_record_buffers(record_name: str) -> None:
        record_clip_embeddings.setdefault(record_name, [])
        record_clip_record_names.setdefault(record_name, [])
        record_clip_segment_ids.setdefault(record_name, [])
        record_clip_labels.setdefault(record_name, [])
        record_clip_label_text.setdefault(record_name, [])
        record_clip_rhythms.setdefault(record_name, [])
        record_clip_indices.setdefault(record_name, [])
        record_clip_start_frames.setdefault(record_name, [])
        record_clip_end_frames.setdefault(record_name, [])
        record_segment_num_clips.setdefault(record_name, [])
        record_segment_record_names.setdefault(record_name, [])
        record_segment_ids.setdefault(record_name, [])
        record_segment_labels.setdefault(record_name, [])
        record_segment_label_text.setdefault(record_name, [])
        record_segment_rhythms.setdefault(record_name, [])
        record_segment_num_frames.setdefault(record_name, [])
        record_expected_clips.setdefault(record_name, 0)
        record_emitted_clips.setdefault(record_name, 0)

    def emit_record(record_name: str) -> None:
        record_embeddings = np.concatenate(record_clip_embeddings[record_name], axis=0)
        record_meta = {
            "clip_record_names": np.array(record_clip_record_names[record_name]),
            "clip_segment_ids": np.array(record_clip_segment_ids[record_name], dtype=np.int32),
            "clip_labels": np.array(record_clip_labels[record_name], dtype=np.int32),
            "clip_label_text": np.array(record_clip_label_text[record_name]),
            "clip_rhythms": np.array(record_clip_rhythms[record_name]),
            "clip_indices": np.array(record_clip_indices[record_name], dtype=np.int32),
            "clip_start_frames": np.array(record_clip_start_frames[record_name], dtype=np.int32),
            "clip_end_frames": np.array(record_clip_end_frames[record_name], dtype=np.int32),
            "segment_record_names": np.array(record_segment_record_names[record_name]),
            "segment_ids": np.array(record_segment_ids[record_name], dtype=np.int32),
            "segment_labels": np.array(record_segment_labels[record_name], dtype=np.int32),
            "segment_label_text": np.array(record_segment_label_text[record_name]),
            "segment_rhythms": np.array(record_segment_rhythms[record_name]),
            "segment_num_frames": np.array(record_segment_num_frames[record_name], dtype=np.int32),
            "segment_num_clips": np.array(record_segment_num_clips[record_name], dtype=np.int32),
        }

        clip_embeddings.append(record_embeddings)
        for key, value in record_meta.items():
            meta_chunks.setdefault(key, []).append(value)

        if on_record_complete is not None:
            on_record_complete(record_name, record_embeddings, record_meta)

        emitted_records.add(record_name)
        del record_clip_embeddings[record_name]
        del record_clip_record_names[record_name]
        del record_clip_segment_ids[record_name]
        del record_clip_labels[record_name]
        del record_clip_label_text[record_name]
        del record_clip_rhythms[record_name]
        del record_clip_indices[record_name]
        del record_clip_start_frames[record_name]
        del record_clip_end_frames[record_name]
        del record_segment_num_clips[record_name]
        del record_segment_record_names[record_name]
        del record_segment_ids[record_name]
        del record_segment_labels[record_name]
        del record_segment_label_text[record_name]
        del record_segment_rhythms[record_name]
        del record_segment_num_frames[record_name]
        del record_expected_clips[record_name]
        del record_emitted_clips[record_name]

    def maybe_emit_completed_records() -> None:
        for record_name in list(closed_records):
            if record_name in emitted_records:
                closed_records.discard(record_name)
                continue
            if record_emitted_clips[record_name] == record_expected_clips[record_name]:
                emit_record(record_name)
                closed_records.discard(record_name)

    def flush_batch() -> None:
        nonlocal batch_clips, batch_meta
        if not batch_clips:
            return
        inputs = processor(batch_clips, return_tensors="pt").to(device)
        outputs = model(**inputs, skip_predictor=True)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
        for embedding, meta in zip(batch_embeddings, batch_meta):
            record_name, segment_id, label_id, label_text, rhythm, clip_index, start_frame, end_frame = meta
            init_record_buffers(record_name)
            record_clip_embeddings[record_name].append(embedding[np.newaxis, :])
            record_clip_record_names[record_name].append(record_name)
            record_clip_segment_ids[record_name].append(segment_id)
            record_clip_labels[record_name].append(label_id)
            record_clip_label_text[record_name].append(label_text)
            record_clip_rhythms[record_name].append(rhythm)
            record_clip_indices[record_name].append(clip_index)
            record_clip_start_frames[record_name].append(start_frame)
            record_clip_end_frames[record_name].append(end_frame)
            record_emitted_clips[record_name] += 1

        batch_clips = []
        batch_meta = []
        maybe_emit_completed_records()

    with torch.no_grad():
        previous_record_name: str | None = None
        for segment_index, example in enumerate(examples, start=1):
            if previous_record_name is not None and example.record_name != previous_record_name:
                closed_records.add(previous_record_name)
                maybe_emit_completed_records()
            previous_record_name = example.record_name
            init_record_buffers(example.record_name)

            frames = load_segment_frames(example.frames_dir)
            clips_with_meta = build_fixed_length_clips_with_metadata(
                frames,
                target_num_frames=target_num_frames,
                stride=clip_stride,
            )

            segment_clip_count = len(clips_with_meta)
            record_expected_clips[example.record_name] += segment_clip_count
            for clip_index, (clip_frames, start_frame, end_frame) in enumerate(clips_with_meta):
                batch_clips.append(clip_frames)
                batch_meta.append(
                    (
                        example.record_name,
                        example.segment_id,
                        example.label_id,
                        example.label_text,
                        example.rhythm,
                        clip_index,
                        start_frame,
                        end_frame,
                    )
                )
                if len(batch_clips) >= batch_size:
                    flush_batch()

            record_segment_num_clips[example.record_name].append(segment_clip_count)
            record_segment_record_names[example.record_name].append(example.record_name)
            record_segment_ids[example.record_name].append(example.segment_id)
            record_segment_labels[example.record_name].append(example.label_id)
            record_segment_label_text[example.record_name].append(example.label_text)
            record_segment_rhythms[example.record_name].append(example.rhythm)
            record_segment_num_frames[example.record_name].append(example.num_frames)

            print(
                f"[{segment_index}/{len(examples)}] "
                f"record={example.record_name} segment={example.segment_id} "
                f"frames={example.num_frames} clips={segment_clip_count}"
            )

        if previous_record_name is not None:
            closed_records.add(previous_record_name)
        flush_batch()
        maybe_emit_completed_records()

    if not clip_embeddings:
        raise ValueError("No clip embeddings were generated.")

    meta = {key: np.concatenate(chunks, axis=0) for key, chunks in meta_chunks.items()}
    return np.concatenate(clip_embeddings, axis=0), meta


def save_embedding_cache(cache_path: Path, embeddings: np.ndarray, examples: list[SegmentExample]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        record_names=np.array([example.record_name for example in examples]),
        segment_ids=np.array([example.segment_id for example in examples], dtype=np.int32),
        labels=np.array([example.label_id for example in examples], dtype=np.int32),
        label_text=np.array([example.label_text for example in examples]),
        rhythms=np.array([example.rhythm for example in examples]),
    )


def load_embedding_cache(cache_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache = np.load(cache_path, allow_pickle=False)
    meta = {
        "record_names": cache["record_names"],
        "segment_ids": cache["segment_ids"],
        "labels": cache["labels"],
        "label_text": cache["label_text"],
        "rhythms": cache["rhythms"],
    }
    return cache["embeddings"], meta


def save_clip_embedding_cache(cache_path: Path, embeddings: np.ndarray, meta: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, **meta)


def load_clip_embedding_cache(cache_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache = np.load(cache_path, allow_pickle=False)
    meta = {key: cache[key] for key in cache.files if key != "embeddings"}
    return cache["embeddings"], meta


def cache_matches_examples(cache_meta: dict[str, np.ndarray], examples: list[SegmentExample]) -> bool:
    if len(cache_meta["segment_ids"]) != len(examples):
        return False

    cache_pairs = list(zip(cache_meta["record_names"].tolist(), cache_meta["segment_ids"].tolist()))
    example_pairs = [(example.record_name, example.segment_id) for example in examples]
    return cache_pairs == example_pairs


def clip_cache_matches_examples(cache_meta: dict[str, np.ndarray], examples: list[SegmentExample]) -> bool:
    if "segment_record_names" not in cache_meta or "segment_ids" not in cache_meta:
        return False
    if len(cache_meta["segment_ids"]) != len(examples):
        return False

    cache_pairs = list(zip(cache_meta["segment_record_names"].tolist(), cache_meta["segment_ids"].tolist()))
    example_pairs = [(example.record_name, example.segment_id) for example in examples]
    return cache_pairs == example_pairs


def record_cache_path(cache_root: Path, record_name: str) -> Path:
    return cache_root / f"{record_name}.npz"


def save_record_embedding_cache(cache_root: Path, record_name: str, embeddings: np.ndarray, examples: list[SegmentExample]) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = record_cache_path(cache_root, record_name)
    save_embedding_cache(cache_path, embeddings, examples)
    return cache_path


def load_record_embedding_cache(cache_root: Path, record_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return load_embedding_cache(record_cache_path(cache_root, record_name))


def load_record_clip_embedding_cache(cache_root: Path, record_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return load_clip_embedding_cache(record_cache_path(cache_root, record_name))


def group_examples_by_record(examples: list[SegmentExample]) -> dict[str, list[SegmentExample]]:
    grouped: dict[str, list[SegmentExample]] = {}
    for example in examples:
        grouped.setdefault(example.record_name, []).append(example)
    return grouped


def filter_record_clip_embedding_cache(
    record_name: str,
    embeddings: np.ndarray,
    meta: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    clip_mask = meta["clip_record_names"] == record_name
    segment_mask = meta["segment_record_names"] == record_name
    record_meta = {
        "clip_record_names": meta["clip_record_names"][clip_mask],
        "clip_segment_ids": meta["clip_segment_ids"][clip_mask],
        "clip_labels": meta["clip_labels"][clip_mask],
        "clip_label_text": meta["clip_label_text"][clip_mask],
        "clip_rhythms": meta["clip_rhythms"][clip_mask],
        "clip_indices": meta["clip_indices"][clip_mask],
        "clip_start_frames": meta["clip_start_frames"][clip_mask],
        "clip_end_frames": meta["clip_end_frames"][clip_mask],
        "segment_record_names": meta["segment_record_names"][segment_mask],
        "segment_ids": meta["segment_ids"][segment_mask],
        "segment_labels": meta["segment_labels"][segment_mask],
        "segment_label_text": meta["segment_label_text"][segment_mask],
        "segment_rhythms": meta["segment_rhythms"][segment_mask],
        "segment_num_frames": meta["segment_num_frames"][segment_mask],
        "segment_num_clips": meta["segment_num_clips"][segment_mask],
    }
    return embeddings[clip_mask], record_meta


def ensure_embedding_cache(
    *,
    dataset_root: Path,
    embedding_cache: Path,
    selected_records: set[str] | None,
    max_segments: int | None,
    model_name: str,
    device_arg: str,
    batch_size: int,
    target_num_frames: int,
    clip_stride: int,
    force_recompute: bool,
) -> tuple[list[SegmentExample], np.ndarray, np.ndarray, np.ndarray]:
    examples = load_segment_examples(
        dataset_root=dataset_root,
        selected_records=selected_records,
        max_segments=max_segments,
    )

    if uses_legacy_single_file_cache(embedding_cache):
        if embedding_cache.exists() and not force_recompute:
            embeddings, cache_meta = load_embedding_cache(embedding_cache)
            if cache_matches_examples(cache_meta, examples):
                labels = cache_meta["labels"].astype(np.int32)
                record_names = cache_meta["record_names"]
                print(f"Loaded cached embeddings from {embedding_cache} with {len(labels)} segments.")
                return examples, embeddings, labels, record_names

            print(f"Cache at {embedding_cache} does not match the selected segments. Recomputing embeddings.")

        device = resolve_device(device_arg)
        print(f"Using device: {device}")
        embeddings = embed_segments(
            examples,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            target_num_frames=target_num_frames,
            clip_stride=clip_stride,
        )
        save_embedding_cache(embedding_cache, embeddings, examples)
        labels = np.array([example.label_id for example in examples], dtype=np.int32)
        record_names = np.array([example.record_name for example in examples])
        print(f"Saved embedding cache to {embedding_cache}")
        return examples, embeddings, labels, record_names

    cache_root = embedding_cache
    grouped_examples = group_examples_by_record(examples)
    device: str | None = None
    ordered_examples: list[SegmentExample] = []
    embeddings_per_record: list[np.ndarray] = []
    labels_per_record: list[np.ndarray] = []
    record_names_per_record: list[np.ndarray] = []

    for record_name in sorted(grouped_examples):
        record_examples = grouped_examples[record_name]
        cache_path = record_cache_path(cache_root, record_name)

        if cache_path.exists() and not force_recompute:
            record_embeddings, cache_meta = load_record_embedding_cache(cache_root, record_name)
            if cache_matches_examples(cache_meta, record_examples):
                print(f"Loaded cached embeddings for record {record_name} from {cache_path}")
            else:
                print(f"Cache at {cache_path} does not match record {record_name}. Recomputing embeddings.")
                if device is None:
                    device = resolve_device(device_arg)
                    print(f"Using device: {device}")
                record_embeddings = embed_segments(
                    record_examples,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    target_num_frames=target_num_frames,
                    clip_stride=clip_stride,
                )
                save_record_embedding_cache(cache_root, record_name, record_embeddings, record_examples)
                print(f"Saved embedding cache for record {record_name} to {cache_path}")
        else:
            if device is None:
                device = resolve_device(device_arg)
                print(f"Using device: {device}")
            record_embeddings = embed_segments(
                record_examples,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                target_num_frames=target_num_frames,
                clip_stride=clip_stride,
            )
            save_record_embedding_cache(cache_root, record_name, record_embeddings, record_examples)
            print(f"Saved embedding cache for record {record_name} to {cache_path}")

        ordered_examples.extend(record_examples)
        embeddings_per_record.append(record_embeddings)
        labels_per_record.append(np.array([example.label_id for example in record_examples], dtype=np.int32))
        record_names_per_record.append(np.array([record_name] * len(record_examples)))

    embeddings = np.concatenate(embeddings_per_record, axis=0)
    labels = np.concatenate(labels_per_record, axis=0)
    record_names = np.concatenate(record_names_per_record, axis=0)
    return ordered_examples, embeddings, labels, record_names


def ensure_clip_embedding_cache(
    *,
    dataset_root: Path,
    embedding_cache: Path,
    selected_records: set[str] | None,
    max_segments: int | None,
    model_name: str,
    device_arg: str,
    batch_size: int,
    target_num_frames: int,
    clip_stride: int,
    force_recompute: bool,
) -> tuple[list[SegmentExample], np.ndarray, dict[str, np.ndarray]]:
    examples = load_segment_examples(
        dataset_root=dataset_root,
        selected_records=selected_records,
        max_segments=max_segments,
    )

    if uses_legacy_single_file_cache(embedding_cache):
        if embedding_cache.exists() and not force_recompute:
            embeddings, cache_meta = load_clip_embedding_cache(embedding_cache)
            if clip_cache_matches_examples(cache_meta, examples):
                print(f"Loaded cached clip embeddings from {embedding_cache} with {len(cache_meta['segment_ids'])} segments.")
                return examples, embeddings, cache_meta
            print(f"Cache at {embedding_cache} does not match the selected segments. Recomputing clip embeddings.")

        device = resolve_device(device_arg)
        print(f"Using device: {device}")
        embeddings, cache_meta = embed_segment_clips(
            examples,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            target_num_frames=target_num_frames,
            clip_stride=clip_stride,
        )
        save_clip_embedding_cache(embedding_cache, embeddings, cache_meta)
        print(f"Saved clip embedding cache to {embedding_cache}")
        return examples, embeddings, cache_meta

    cache_root = embedding_cache
    grouped_examples = group_examples_by_record(examples)
    device: str | None = None
    record_results: dict[str, tuple[list[SegmentExample], np.ndarray, dict[str, np.ndarray]]] = {}

    records_to_compute: list[str] = []
    examples_to_compute: list[SegmentExample] = []

    for record_name in sorted(grouped_examples):
        record_examples = grouped_examples[record_name]
        cache_path = record_cache_path(cache_root, record_name)

        if cache_path.exists() and not force_recompute:
            record_embeddings, record_meta = load_record_clip_embedding_cache(cache_root, record_name)
            if clip_cache_matches_examples(record_meta, record_examples):
                record_results[record_name] = (record_examples, record_embeddings, record_meta)
            else:
                print(f"Cache at {cache_path} does not match record {record_name}. Recomputing clip embeddings.")
                records_to_compute.append(record_name)
                examples_to_compute.extend(record_examples)
        else:
            records_to_compute.append(record_name)
            examples_to_compute.extend(record_examples)

    if examples_to_compute:
        if device is None:
            device = resolve_device(device_arg)
            print(f"Using device: {device}")

        def save_completed_record(
            record_name: str,
            record_embeddings: np.ndarray,
            record_meta: dict[str, np.ndarray],
        ) -> None:
            record_examples = grouped_examples[record_name]
            cache_path = record_cache_path(cache_root, record_name)
            save_clip_embedding_cache(cache_path, record_embeddings, record_meta)
            print(f"Saved clip embedding cache for record {record_name} to {cache_path}")
            record_results[record_name] = (record_examples, record_embeddings, record_meta)

        embed_segment_clips(
            examples_to_compute,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            target_num_frames=target_num_frames,
            clip_stride=clip_stride,
            on_record_complete=save_completed_record,
        )

    ordered_examples: list[SegmentExample] = []
    embeddings_per_record: list[np.ndarray] = []
    meta_chunks: dict[str, list[np.ndarray]] = {}
    for record_name in sorted(grouped_examples):
        record_examples, record_embeddings, record_meta = record_results[record_name]
        ordered_examples.extend(record_examples)
        embeddings_per_record.append(record_embeddings)
        for key, value in record_meta.items():
            meta_chunks.setdefault(key, []).append(value)

    embeddings = np.concatenate(embeddings_per_record, axis=0)
    combined_meta = {key: np.concatenate(chunks, axis=0) for key, chunks in meta_chunks.items()}
    return ordered_examples, embeddings, combined_meta
