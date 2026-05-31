from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate exported ECG frame PNGs for one or more records into displayable row images."
    )
    parser.add_argument(
        "records",
        nargs="+",
        help="Record ID/name(s) to plot, for example 102 or 102 10045 15970.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("src/data/ptbxl_vjepa_frames"),
        help="Root directory containing exported record frame folders.",
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        default=0,
        help="Segment ID to plot. PTB-XL exports usually use segment 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output image path. If --frame-indices is used, this is treated as a filename prefix "
            "and one file per frame is written."
        ),
    )
    parser.add_argument(
        "--max-width",
        type=int,
        help="Optional maximum output width. Preserves aspect ratio if resizing is needed.",
    )
    parser.add_argument(
        "--frame-indices",
        nargs="+",
        type=int,
        help="Plot selected single frame indices instead of concatenating the whole record.",
    )
    parser.add_argument(
        "--black",
        action="store_true",
        help="Render ECG curves in black instead of class colors.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="ECG curve thickness in pixels after binarization.",
    )

    return parser.parse_args()

def load_frames(frames_dir: Path) -> list[np.ndarray]:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise ValueError(f"No frame PNG files found in {frames_dir}.")

    frames: list[np.ndarray] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Failed to read frame {frame_path}.")
        frames.append(frame)
    return frames


def resize_to_max_width(image: np.ndarray, max_width: int | None) -> np.ndarray:
    if max_width is None or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    new_size = (max_width, max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def make_colored_on_white(image: np.ndarray, label_text: str, thickness: int, black: bool) -> np.ndarray:
    if thickness < 1:
        raise ValueError("--thickness must be at least 1.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    if thickness > 1:
        kernel = np.ones((thickness, thickness), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    output = np.full_like(image, 255)
    if black:
        color = (0, 0, 0)
    else:
        color = (0, 160, 0) if label_text == "normal rhythm" else (0, 0, 220)
    output[mask > 0] = color
    return output


def build_record_row(
    dataset_root: Path,
    record: str,
    segment_id: int,
    max_width: int | None,
    thickness: int,
    black: bool,
) -> tuple[np.ndarray, int]:
    segment_dir = dataset_root / str(record) / f"segment_{segment_id:04d}"
    metadata_path = segment_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing segment metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    frames_dir = dataset_root / metadata["frames_dir"]
    label_text = str(metadata["class"])
    frames = load_frames(frames_dir)

    heights = {frame.shape[0] for frame in frames}
    widths = {frame.shape[1] for frame in frames}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError(f"Expected all frames to have the same size, got heights={heights}, widths={widths}.")

    concatenated = np.concatenate(frames, axis=1)
    concatenated = make_colored_on_white(concatenated, label_text, thickness, black)
    concatenated = resize_to_max_width(concatenated, max_width)
    return concatenated, len(frames)


def build_single_frames(
    dataset_root: Path,
    record: str,
    segment_id: int,
    frame_indices: list[int],
    thickness: int,
    black: bool,
) -> list[tuple[int, np.ndarray]]:
    segment_dir = dataset_root / str(record) / f"segment_{segment_id:04d}"
    metadata_path = segment_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing segment metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    frames_dir = dataset_root / metadata["frames_dir"]
    label_text = str(metadata["class"])
    frames = load_frames(frames_dir)

    selected_frames: list[tuple[int, np.ndarray]] = []
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(frames):
            raise IndexError(f"Frame index {frame_index} is out of range for record {record}; valid range is 0-{len(frames) - 1}.")
        selected_frames.append((frame_index, make_colored_on_white(frames[frame_index], label_text, thickness, black)))
    return selected_frames


def output_path_for_frame(output: Path, frame_index: int) -> Path:
    suffix = output.suffix or ".png"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_frame_{frame_index:04d}{suffix}")


def pad_rows_to_same_width(rows: list[np.ndarray]) -> list[np.ndarray]:
    max_width = max(row.shape[1] for row in rows)
    padded_rows: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] == max_width:
            padded_rows.append(row)
            continue
        pad_width = max_width - row.shape[1]
        padding = np.full((row.shape[0], pad_width, row.shape[2]), 255, dtype=row.dtype)
        padded_rows.append(np.concatenate([row, padding], axis=1))
    return padded_rows


def main() -> None:
    args = parse_args()
    if args.frame_indices is not None:
        if len(args.records) != 1:
            raise ValueError("--frame-indices expects exactly one record.")
        single_frames = build_single_frames(
            args.dataset_root,
            args.records[0],
            args.segment_id,
            args.frame_indices,
            args.thickness,
            args.black,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            written_paths: list[Path] = []
            for frame_index, image in single_frames:
                image = resize_to_max_width(image, args.max_width)
                output_path = output_path_for_frame(args.output, frame_index)
                if not cv2.imwrite(str(output_path), image):
                    raise RuntimeError(f"Failed to write output image to {output_path}")
                written_paths.append(output_path)
            print(f"Saved {len(written_paths)} single-frame image(s): {', '.join(str(path) for path in written_paths)}")
            return

        for frame_index, image in single_frames:
            image = resize_to_max_width(image, args.max_width)
            plt.figure(figsize=(4, 4))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Record {args.records[0]} frame {frame_index:04d}")
            plt.tight_layout()
        plt.show()
        return
    else:
        rows: list[np.ndarray] = []
        frame_counts: list[int] = []
        for record in args.records:
            row, frame_count = build_record_row(
                args.dataset_root,
                record,
                args.segment_id,
                args.max_width,
                args.thickness,
                args.black,
            )
            rows.append(row)
            frame_counts.append(frame_count)

        stacked = np.concatenate(pad_rows_to_same_width(rows), axis=0)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), stacked):
            raise RuntimeError(f"Failed to write output image to {args.output}")
        print(
            f"Saved {len(args.records)} record row(s) to {args.output} with shape {stacked.shape}. "
            f"Frame counts: {dict(zip(args.records, frame_counts))}"
        )
        return

    plt.figure(figsize=(16, max(3, 2 * len(args.records))))
    plt.imshow(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    # plt.title(f"Records {', '.join(args.records)} segment {args.segment_id:04d}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

# %%
