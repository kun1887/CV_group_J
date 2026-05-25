from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from vjepa_embedding_utils import (
    build_fixed_length_clips_with_metadata,
    load_segment_frames,
    resolve_device,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "data" / "ptbxl_vjepa_frames"
DEFAULT_PARAMETERS = SCRIPT_DIR / "data" / "ptbxl_vjepa_linear_probe_fpc16" / "linear_probe_parameters.npz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "ptbxl_vjepa_logistic_patch_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one PTB-XL frame export by label and save a video with V-JEPA patch contribution heatmaps "
            "for the saved logistic not-normal direction."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder containing PTB-XL frame exports.",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=DEFAULT_PARAMETERS,
        help="Saved logistic probe parameters from linear_probe_pooled_clip_embeddings.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the heatmap video and summary JSON are saved.",
    )
    parser.add_argument(
        "--label",
        choices=("normal", "not-normal", "not_normal"),
        required=True,
        help="Record label to sample.",
    )
    parser.add_argument(
        "--record",
        help="Optional explicit PTB-XL ecg_id to visualize instead of random sampling.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/vjepa2-vitl-fpc16-256-ssv2",
        help="Hugging Face V-JEPA model used for fpc16 embeddings.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used for V-JEPA inference.",
    )
    parser.add_argument("--target-num-frames", type=int, default=16, help="Frames per V-JEPA clip.")
    parser.add_argument("--clip-stride", type=int, default=8, help="Stride between clips in frame units.")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of clips per V-JEPA forward pass.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for record sampling.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Maximum heatmap alpha for strongest positive/negative patch contribution.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Output video FPS. Defaults to 1 / seconds_per_frame from the record metadata.",
    )
    return parser.parse_args()


def normalized_label(label: str) -> str:
    return "not-normal rhythm" if label in {"not-normal", "not_normal"} else "normal rhythm"


def natural_key(path: Path) -> tuple[int, int | str]:
    record_name = path.parent.parent.name
    return (0, int(record_name)) if record_name.isdigit() else (1, record_name)


def select_record(dataset_root: Path, label_text: str, record_name: str | None, seed: int) -> tuple[Path, dict]:
    metadata_paths = sorted(dataset_root.glob("*/segment_0000/metadata.json"), key=natural_key)
    matches: list[tuple[Path, dict]] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        current_record = metadata_path.parent.parent.name
        if record_name is not None and current_record != str(record_name):
            continue
        if metadata.get("class") == label_text:
            matches.append((metadata_path, metadata))

    if record_name is not None and not matches:
        raise ValueError(f"Record {record_name} was not found with label {label_text!r}.")
    if not matches:
        raise ValueError(f"No records with label {label_text!r} found under {dataset_root}.")

    rng = random.Random(seed)
    return rng.choice(matches)


def infer_token_grid(num_tokens: int, target_num_frames: int, model: torch.nn.Module) -> tuple[int, int, int]:
    config = model.config
    tubelet_size = int(getattr(config, "tubelet_size", 2))
    patch_size = int(getattr(config, "patch_size", 16))
    image_size = int(getattr(config, "image_size", getattr(config, "crop_size", 256)))
    temporal_grid = max(1, target_num_frames // tubelet_size)
    spatial_grid = image_size // patch_size

    expected_tokens = temporal_grid * spatial_grid * spatial_grid
    if expected_tokens == num_tokens:
        return temporal_grid, spatial_grid, spatial_grid

    inferred_spatial = int(round(np.sqrt(num_tokens / temporal_grid)))
    if temporal_grid * inferred_spatial * inferred_spatial == num_tokens:
        return temporal_grid, inferred_spatial, inferred_spatial

    raise ValueError(
        f"Cannot map {num_tokens} V-JEPA tokens to a regular frame grid. "
        f"Expected {expected_tokens} from temporal={temporal_grid}, spatial={spatial_grid}."
    )


def overlay_signed_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
    scale_max_abs: float,
) -> np.ndarray:
    frame_float = to_white_background_black_trace(frame).astype(np.float32)
    heatmap = heatmap.astype(np.float32)
    if scale_max_abs <= 1e-12:
        return frame.copy()

    normalized = np.clip(heatmap / scale_max_abs, -1.0, 1.0)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    magnitude = np.maximum(positive, negative)

    color = np.zeros_like(frame_float)
    color[..., 0] = 255.0 * positive
    color[..., 2] = 255.0 * negative
    alpha_map = (alpha * magnitude)[..., np.newaxis]
    blended = frame_float * (1.0 - alpha_map) + color * alpha_map
    return np.clip(blended, 0, 255).astype(np.uint8)


def to_white_background_black_trace(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, trace_mask = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)
    output = np.full_like(frame, 255)
    output[trace_mask > 0] = 0
    return output


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def logit_threshold(probability_threshold: float) -> float:
    threshold = float(np.clip(probability_threshold, 1e-8, 1.0 - 1e-8))
    return float(np.log(threshold / (1.0 - threshold)))


def compute_cumulative_scores(
    frame_contributions: np.ndarray,
    raw_intercept: float,
    threshold: float,
) -> dict[str, np.ndarray | float]:
    cumulative_mean = np.cumsum(frame_contributions) / np.arange(1, len(frame_contributions) + 1)
    cumulative_logit = raw_intercept + cumulative_mean
    cumulative_prob = sigmoid(cumulative_logit)
    threshold_logit = logit_threshold(threshold)
    cumulative_margin = cumulative_logit - threshold_logit
    return {
        "cumulative_mean_contribution": cumulative_mean,
        "cumulative_logit": cumulative_logit,
        "cumulative_prob_not_normal": cumulative_prob,
        "cumulative_margin_to_threshold": cumulative_margin,
        "threshold_logit": threshold_logit,
    }


def draw_cumulative_score_bar(
    frame: np.ndarray,
    cumulative_margin: float,
    scale_max_abs: float,
) -> np.ndarray:
    if scale_max_abs <= 1e-12:
        return frame

    output = frame.copy()
    normalized = float(np.clip(cumulative_margin / scale_max_abs, -1.0, 1.0))
    magnitude = abs(normalized)
    if magnitude <= 1e-12:
        return output

    height, width = output.shape[:2]
    bar_height = max(6, int(round(height * 0.035)))
    color = np.array([220, 38, 38], dtype=np.float32) if normalized > 0 else np.array([37, 99, 235], dtype=np.float32)
    alpha = 0.2 + 0.7 * magnitude
    strip = output[:bar_height].astype(np.float32)
    strip = strip * (1.0 - alpha) + color * alpha
    output[:bar_height] = np.clip(strip, 0, 255).astype(np.uint8)
    return output


def compute_frame_heatmaps(
    frames: np.ndarray,
    model_name: str,
    device_arg: str,
    raw_weight: np.ndarray,
    raw_intercept: float,
    target_num_frames: int,
    clip_stride: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    device = torch.device(resolve_device(device_arg))
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoVideoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    model.eval()

    clips_with_meta = build_fixed_length_clips_with_metadata(
        frames,
        target_num_frames=target_num_frames,
        stride=clip_stride,
    )
    frame_heatmaps: np.ndarray | None = None
    frame_counts: np.ndarray | None = None
    token_grid: tuple[int, int, int] | None = None
    distributed_intercept = 0.0
    raw_weight_tensor = torch.from_numpy(raw_weight.astype(np.float32)).to(device)

    with torch.no_grad():
        for batch_start in range(0, len(clips_with_meta), batch_size):
            batch = clips_with_meta[batch_start : batch_start + batch_size]
            batch_clips = [clip for clip, _, _ in batch]
            inputs = processor(batch_clips, return_tensors="pt").to(device)
            outputs = model(**inputs, skip_predictor=True)
            token_embeddings = outputs.last_hidden_state.float()
            contributions = torch.matmul(token_embeddings, raw_weight_tensor).cpu().numpy()

            if token_grid is None:
                token_grid = infer_token_grid(contributions.shape[1], target_num_frames, model)
                _, grid_h, grid_w = token_grid
                frame_heatmaps = np.zeros((len(frames), grid_h, grid_w), dtype=np.float64)
                frame_counts = np.zeros((len(frames), grid_h, grid_w), dtype=np.float64)
                num_record_tokens = len(clips_with_meta) * contributions.shape[1]
                distributed_intercept = raw_intercept / max(num_record_tokens, 1)

            contributions = contributions + distributed_intercept

            temporal_grid, grid_h, grid_w = token_grid
            tubelet_size = max(1, target_num_frames // temporal_grid)

            for clip_contrib, (_, start_frame, _) in zip(contributions, batch):
                clip_grid = clip_contrib.reshape(temporal_grid, grid_h, grid_w)
                for local_frame in range(target_num_frames):
                    frame_index = start_frame + local_frame
                    if frame_index >= len(frames):
                        continue
                    temporal_index = min(local_frame // tubelet_size, temporal_grid - 1)
                    frame_heatmaps[frame_index] += clip_grid[temporal_index]
                    frame_counts[frame_index] += 1.0

    if frame_heatmaps is None or frame_counts is None or token_grid is None:
        raise RuntimeError("No frame heatmaps were computed.")

    frame_heatmaps = frame_heatmaps / np.maximum(frame_counts, 1.0)
    metadata = {
        "num_clips": len(clips_with_meta),
        "token_grid": {
            "temporal": token_grid[0],
            "height": token_grid[1],
            "width": token_grid[2],
        },
        "num_tokens_per_clip": int(np.prod(token_grid)),
        "patch_intercept_mode": "distributed_across_record_tokens",
        "distributed_intercept_per_token": float(raw_intercept / max(len(clips_with_meta) * int(np.prod(token_grid)), 1)),
        "contribution_sign": (
            "positive values support the logistic not-normal direction after adding the distributed intercept share"
        ),
    }
    return frame_heatmaps.astype(np.float32), metadata


def save_video(
    output_path: Path,
    frames: np.ndarray,
    frame_heatmaps: np.ndarray,
    cumulative_margins: np.ndarray,
    fps: float,
    alpha: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}.")

    global_max_abs = float(np.max(np.abs(frame_heatmaps)))
    cumulative_margin_max_abs = float(np.max(np.abs(cumulative_margins)))
    for frame, heatmap, cumulative_margin in zip(frames, frame_heatmaps, cumulative_margins):
        heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        overlaid = overlay_signed_heatmap(frame, heatmap_resized, alpha=alpha, scale_max_abs=global_max_abs)
        overlaid = draw_cumulative_score_bar(overlaid, float(cumulative_margin), cumulative_margin_max_abs)
        writer.write(cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    writer.release()
    return global_max_abs, cumulative_margin_max_abs


def save_frame_contributions(
    path: Path,
    frame_contributions: np.ndarray,
    cumulative_scores: dict[str, np.ndarray | float],
    threshold: float,
    seconds_per_frame: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "start_sec",
                "end_sec",
                "frame_contribution",
                "contribution_direction",
                "cumulative_mean_contribution",
                "cumulative_logit",
                "cumulative_prob_not_normal",
                "threshold",
                "cumulative_margin_to_threshold",
                "cumulative_prediction",
            ],
        )
        writer.writeheader()
        cumulative_mean = np.asarray(cumulative_scores["cumulative_mean_contribution"])
        cumulative_logit = np.asarray(cumulative_scores["cumulative_logit"])
        cumulative_prob = np.asarray(cumulative_scores["cumulative_prob_not_normal"])
        cumulative_margin = np.asarray(cumulative_scores["cumulative_margin_to_threshold"])
        for frame_index, contribution in enumerate(frame_contributions):
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "start_sec": frame_index * seconds_per_frame,
                    "end_sec": (frame_index + 1) * seconds_per_frame,
                    "frame_contribution": float(contribution),
                    "contribution_direction": "not-normal" if contribution > 0 else "normal" if contribution < 0 else "neutral",
                    "cumulative_mean_contribution": float(cumulative_mean[frame_index]),
                    "cumulative_logit": float(cumulative_logit[frame_index]),
                    "cumulative_prob_not_normal": float(cumulative_prob[frame_index]),
                    "threshold": float(threshold),
                    "cumulative_margin_to_threshold": float(cumulative_margin[frame_index]),
                    "cumulative_prediction": "not-normal" if cumulative_prob[frame_index] >= threshold else "normal",
                }
            )


def main() -> None:
    args = parse_args()
    label_text = normalized_label(args.label)
    metadata_path, metadata = select_record(args.dataset_root, label_text, args.record, args.seed)
    frames_dir = args.dataset_root / metadata["frames_dir"]
    frames = load_segment_frames(frames_dir)

    params = np.load(args.parameters, allow_pickle=False)
    raw_weight = params["raw_weight"].astype(np.float64)
    raw_intercept = float(params["raw_intercept"])
    threshold = float(params["threshold"])

    frame_heatmaps, heatmap_metadata = compute_frame_heatmaps(
        frames=frames,
        model_name=args.model_name,
        device_arg=args.device,
        raw_weight=raw_weight,
        raw_intercept=raw_intercept,
        target_num_frames=args.target_num_frames,
        clip_stride=args.clip_stride,
        batch_size=args.batch_size,
    )
    fps = args.fps if args.fps is not None else 1.0 / float(metadata["seconds_per_frame"])
    seconds_per_frame = float(metadata["seconds_per_frame"])
    record_name = metadata["record_name"]
    safe_label = "not_normal" if label_text == "not-normal rhythm" else "normal"
    output_path = args.output_dir / f"record_{record_name}_{safe_label}_logistic_patch_contributions.mp4"
    frame_contributions = frame_heatmaps.mean(axis=(1, 2))
    cumulative_scores = compute_cumulative_scores(frame_contributions, raw_intercept, threshold)
    cumulative_margins = np.asarray(cumulative_scores["cumulative_margin_to_threshold"])
    global_heatmap_max_abs, cumulative_margin_max_abs = save_video(
        output_path,
        frames,
        frame_heatmaps,
        cumulative_margins,
        fps=fps,
        alpha=args.alpha,
    )
    frame_contributions_path = output_path.with_name(f"{output_path.stem}_frame_contributions.csv")
    save_frame_contributions(frame_contributions_path, frame_contributions, cumulative_scores, threshold, seconds_per_frame)

    summary = {
        "record_name": record_name,
        "label": label_text,
        "rhythm": metadata["rhythm"],
        "metadata_path": str(metadata_path),
        "frames_dir": str(frames_dir),
        "output_video": str(output_path),
        "parameters": str(args.parameters),
        "model_name": args.model_name,
        "num_frames": int(len(frames)),
        "fps": float(fps),
        "target_num_frames": int(args.target_num_frames),
        "clip_stride": int(args.clip_stride),
        "raw_intercept": raw_intercept,
        "threshold": threshold,
        "heatmap_color_scale": "global_max_abs_across_record_frames",
        "global_heatmap_max_abs": global_heatmap_max_abs,
        "cumulative_score_bar": {
            "meaning": (
                "red means the cumulative record score up to this frame is above the tuned not-normal threshold; "
                "blue means it is below the threshold"
            ),
            "cumulative_score": "raw_intercept + mean(frame_contributions_seen_so_far)",
            "cumulative_margin_max_abs": cumulative_margin_max_abs,
            "csv": str(frame_contributions_path),
        },
        "mean_frame_contribution": float(frame_contributions.mean()),
        "min_frame_contribution": float(frame_contributions.min()),
        "max_frame_contribution": float(frame_contributions.max()),
        "final_cumulative_logit": float(np.asarray(cumulative_scores["cumulative_logit"])[-1]),
        "final_cumulative_prob_not_normal": float(np.asarray(cumulative_scores["cumulative_prob_not_normal"])[-1]),
        "final_cumulative_prediction": (
            "not-normal" if np.asarray(cumulative_scores["cumulative_prob_not_normal"])[-1] >= threshold else "normal"
        ),
        "threshold_logit": float(cumulative_scores["threshold_logit"]),
        **heatmap_metadata,
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Record: {record_name} ({label_text})")
    print(f"Video: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Frame contributions: {frame_contributions_path}")


if __name__ == "__main__":
    main()
