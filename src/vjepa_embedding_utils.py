from __future__ import annotations

import json
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
    clip_record_names: list[str] = []
    clip_segment_ids: list[int] = []
    clip_labels: list[int] = []
    clip_label_text: list[str] = []
    clip_rhythms: list[str] = []
    clip_indices: list[int] = []
    clip_start_frames: list[int] = []
    clip_end_frames: list[int] = []
    segment_num_clips: list[int] = []
    segment_record_names: list[str] = []
    segment_ids: list[int] = []
    segment_labels: list[int] = []
    segment_label_text: list[str] = []
    segment_rhythms: list[str] = []
    segment_num_frames: list[int] = []

    with torch.no_grad():
        for segment_index, example in enumerate(examples, start=1):
            frames = load_segment_frames(example.frames_dir)
            clips_with_meta = build_fixed_length_clips_with_metadata(
                frames,
                target_num_frames=target_num_frames,
                stride=clip_stride,
            )

            segment_clip_count = 0
            for batch_start in range(0, len(clips_with_meta), batch_size):
                batch_items = clips_with_meta[batch_start : batch_start + batch_size]
                batch_clips = [item[0] for item in batch_items]
                inputs = processor(batch_clips, return_tensors="pt").to(device)
                outputs = model(**inputs, skip_predictor=True)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
                clip_embeddings.append(batch_embeddings)

                for batch_offset, (_, start_frame, end_frame) in enumerate(batch_items):
                    clip_index = batch_start + batch_offset
                    clip_record_names.append(example.record_name)
                    clip_segment_ids.append(example.segment_id)
                    clip_labels.append(example.label_id)
                    clip_label_text.append(example.label_text)
                    clip_rhythms.append(example.rhythm)
                    clip_indices.append(clip_index)
                    clip_start_frames.append(start_frame)
                    clip_end_frames.append(end_frame)
                    segment_clip_count += 1

            segment_num_clips.append(segment_clip_count)
            segment_record_names.append(example.record_name)
            segment_ids.append(example.segment_id)
            segment_labels.append(example.label_id)
            segment_label_text.append(example.label_text)
            segment_rhythms.append(example.rhythm)
            segment_num_frames.append(example.num_frames)

            print(
                f"[{segment_index}/{len(examples)}] "
                f"record={example.record_name} segment={example.segment_id} "
                f"frames={example.num_frames} clips={segment_clip_count}"
            )

    if not clip_embeddings:
        raise ValueError("No clip embeddings were generated.")

    meta = {
        "clip_record_names": np.array(clip_record_names),
        "clip_segment_ids": np.array(clip_segment_ids, dtype=np.int32),
        "clip_labels": np.array(clip_labels, dtype=np.int32),
        "clip_label_text": np.array(clip_label_text),
        "clip_rhythms": np.array(clip_rhythms),
        "clip_indices": np.array(clip_indices, dtype=np.int32),
        "clip_start_frames": np.array(clip_start_frames, dtype=np.int32),
        "clip_end_frames": np.array(clip_end_frames, dtype=np.int32),
        "segment_record_names": np.array(segment_record_names),
        "segment_ids": np.array(segment_ids, dtype=np.int32),
        "segment_labels": np.array(segment_labels, dtype=np.int32),
        "segment_label_text": np.array(segment_label_text),
        "segment_rhythms": np.array(segment_rhythms),
        "segment_num_frames": np.array(segment_num_frames, dtype=np.int32),
        "segment_num_clips": np.array(segment_num_clips, dtype=np.int32),
    }
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
    ordered_examples: list[SegmentExample] = []
    embeddings_per_record: list[np.ndarray] = []
    meta_chunks: dict[str, list[np.ndarray]] = {}

    for record_name in sorted(grouped_examples):
        record_examples = grouped_examples[record_name]
        cache_path = record_cache_path(cache_root, record_name)

        if cache_path.exists() and not force_recompute:
            record_embeddings, record_meta = load_record_clip_embedding_cache(cache_root, record_name)
            if clip_cache_matches_examples(record_meta, record_examples):
                print(f"Loaded cached clip embeddings for record {record_name} from {cache_path}")
            else:
                print(f"Cache at {cache_path} does not match record {record_name}. Recomputing clip embeddings.")
                if device is None:
                    device = resolve_device(device_arg)
                    print(f"Using device: {device}")
                record_embeddings, record_meta = embed_segment_clips(
                    record_examples,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    target_num_frames=target_num_frames,
                    clip_stride=clip_stride,
                )
                save_clip_embedding_cache(cache_path, record_embeddings, record_meta)
                print(f"Saved clip embedding cache for record {record_name} to {cache_path}")
        else:
            if device is None:
                device = resolve_device(device_arg)
                print(f"Using device: {device}")
            record_embeddings, record_meta = embed_segment_clips(
                record_examples,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                target_num_frames=target_num_frames,
                clip_stride=clip_stride,
            )
            save_clip_embedding_cache(cache_path, record_embeddings, record_meta)
            print(f"Saved clip embedding cache for record {record_name} to {cache_path}")

        ordered_examples.extend(record_examples)
        embeddings_per_record.append(record_embeddings)
        for key, value in record_meta.items():
            meta_chunks.setdefault(key, []).append(value)

    embeddings = np.concatenate(embeddings_per_record, axis=0)
    combined_meta = {key: np.concatenate(chunks, axis=0) for key, chunks in meta_chunks.items()}
    return ordered_examples, embeddings, combined_meta
