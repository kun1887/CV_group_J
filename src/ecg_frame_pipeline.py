from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import wfdb


@dataclass(frozen=True)
class ExportConfig:
    db_name: str = "mitdb"
    annotation_extension: str = "atr"
    lead: int = 0
    seconds_per_frame: float = 1.0
    image_size: int = 224
    pad: int = 18
    line_thickness: int = 2
    draw_grid: bool = True
    grid_step: int = 28
    max_duration_sec: float | None = None
    overwrite: bool = False
    normal_rhythms: tuple[str, ...] = ("(N",)
    ignore_symbols: tuple[str, ...] = ("+",)


def clean_aux_note(aux: object) -> str:
    return str(aux).replace("\x00", "").strip()


def resolve_record_names(
    db_name: str,
    selected_records: Iterable[str] | None = None,
    first_n: int | None = None,
) -> list[str]:
    if selected_records:
        return [str(record) for record in selected_records]

    all_records = list(wfdb.get_record_list(db_name))
    if first_n is not None:
        if first_n <= 0:
            raise ValueError("--first-n must be a positive integer.")
        return all_records[:first_n]
    return all_records


def record_files_exist(data_dir: Path, record_name: str, annotation_extension: str) -> bool:
    required_suffixes = (".dat", ".hea", f".{annotation_extension}")
    return all((data_dir / f"{record_name}{suffix}").exists() for suffix in required_suffixes)


def ensure_records_available(
    db_name: str,
    data_dir: Path,
    record_names: Iterable[str],
    annotation_extension: str,
) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    missing_records = [
        record_name
        for record_name in record_names
        if not record_files_exist(data_dir, record_name, annotation_extension)
    ]
    if missing_records:
        wfdb.dl_database(db_name, dl_dir=str(data_dir), records=missing_records)
    return list(record_names)


def load_record_bundle(data_dir: Path, record_name: str, annotation_extension: str) -> tuple[wfdb.Record, wfdb.Annotation]:
    record_path = str(data_dir / record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, annotation_extension)
    return record, annotation


def build_segments(
    record_name: str,
    record: wfdb.Record,
    annotation: wfdb.Annotation,
    config: ExportConfig,
) -> list[dict]:
    signal = record.p_signal
    fs = float(record.fs)
    rhythm_rows: list[dict] = []

    for sample, aux in zip(annotation.sample, annotation.aux_note):
        rhythm = clean_aux_note(aux)
        if rhythm.startswith("("):
            rhythm_rows.append(
                {
                    "start_sample": int(sample),
                    "start_sec": float(sample / fs),
                    "rhythm": rhythm,
                }
            )

    if not rhythm_rows:
        return []

    normal_rhythms = set(config.normal_rhythms)
    ann_samples = np.asarray(annotation.sample)
    ann_symbols = np.asarray(annotation.symbol)
    ignore_symbols = set(config.ignore_symbols)
    segments: list[dict] = []

    for index, current in enumerate(rhythm_rows):
        next_start = rhythm_rows[index + 1]["start_sample"] if index + 1 < len(rhythm_rows) else signal.shape[0]
        start_sample = current["start_sample"]
        end_sample = int(next_start)

        if config.max_duration_sec is not None:
            max_samples = int(round(config.max_duration_sec * fs))
            end_sample = min(end_sample, start_sample + max_samples)

        if end_sample <= start_sample:
            continue

        in_segment = (ann_samples >= start_sample) & (ann_samples < end_sample)
        segment_symbols = ann_symbols[in_segment]
        beat_symbols = segment_symbols[~np.isin(segment_symbols, list(ignore_symbols))]
        segment_class = "normal rhythm" if current["rhythm"] in normal_rhythms else "not-normal rhythm"

        segments.append(
            {
                "segment_id": len(segments),
                "record_name": record_name,
                "lead_index": int(config.lead),
                "lead_name": record.sig_name[config.lead],
                "rhythm": current["rhythm"],
                "class": segment_class,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_sec": float(start_sample / fs),
                "end_sec": float(end_sample / fs),
                "duration_sec": float((end_sample - start_sample) / fs),
                "num_beats": int(len(beat_symbols)),
                "num_normal_beats": int(np.sum(beat_symbols == "N")),
                "num_abnormal_beats": int(np.sum(beat_symbols != "N")),
            }
        )

    return segments


def render_ecg_chunk_global_scale(
    chunk: np.ndarray,
    *,
    global_min: float,
    global_max: float,
    expected_num_samples: int,
    image_size: int,
    pad: int,
    line_thickness: int,
    draw_grid: bool,
    grid_step: int,
) -> np.ndarray:
    frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    if draw_grid:
        for x in range(0, image_size, grid_step):
            cv2.line(frame, (x, 0), (x, image_size - 1), (28, 28, 28), 1)
        for y in range(0, image_size, grid_step):
            cv2.line(frame, (0, y), (image_size - 1, y), (28, 28, 28), 1)

    chunk = np.asarray(chunk, dtype=np.float32)
    if len(chunk) < 2:
        return frame

    full_x = np.linspace(0, image_size - 1, expected_num_samples).astype(np.int32)
    x_pts = full_x[: len(chunk)]
    usable_height = image_size - 2 * pad
    y_norm = (chunk - global_min) / (global_max - global_min + 1e-6)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    y_pts = (image_size - 1 - (y_norm * usable_height + pad)).astype(np.int32)
    points = np.column_stack([x_pts, y_pts]).astype(np.int32)
    cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 255), thickness=line_thickness)
    return frame


def segment_to_frames(record: wfdb.Record, segment: dict, config: ExportConfig) -> tuple[np.ndarray, dict]:
    signal = record.p_signal
    fs = float(record.fs)
    start_sample = int(segment["start_sample"])
    end_sample = int(segment["end_sample"])
    segment_signal = signal[start_sample:end_sample, config.lead].astype(np.float32)

    if len(segment_signal) == 0:
        raise ValueError(f"Segment {segment['segment_id']} in record {segment['record_name']} contains no samples.")

    samples_per_frame = max(2, int(round(config.seconds_per_frame * fs)))
    num_frames = int(np.ceil(len(segment_signal) / samples_per_frame))
    global_min = float(np.min(segment_signal))
    global_max = float(np.max(segment_signal))

    frames: list[np.ndarray] = []
    valid_samples_per_frame: list[int] = []
    for frame_index in range(num_frames):
        start = frame_index * samples_per_frame
        end = min(start + samples_per_frame, len(segment_signal))
        chunk = segment_signal[start:end]
        valid_samples_per_frame.append(int(len(chunk)))
        frame = render_ecg_chunk_global_scale(
            chunk,
            global_min=global_min,
            global_max=global_max,
            expected_num_samples=samples_per_frame,
            image_size=config.image_size,
            pad=config.pad,
            line_thickness=config.line_thickness,
            draw_grid=config.draw_grid,
            grid_step=config.grid_step,
        )
        frames.append(frame)

    meta = {
        **segment,
        "frames_shape": [num_frames, config.image_size, config.image_size, 3],
        "num_frames": int(num_frames),
        "seconds_per_frame": float(config.seconds_per_frame),
        "samples_per_frame": int(samples_per_frame),
        "valid_samples_per_frame": valid_samples_per_frame,
        "global_min": global_min,
        "global_max": global_max,
    }
    return np.stack(frames, axis=0), meta


def record_output_dir(output_root: Path, record_name: str) -> Path:
    return output_root / record_name


def record_export_complete(output_root: Path, record_name: str) -> bool:
    return (record_output_dir(output_root, record_name) / "export_summary.json").exists()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_segments_csv(path: Path, segments: list[dict]) -> None:
    if not segments:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "segment_id",
                    "record_name",
                    "lead_index",
                    "lead_name",
                    "rhythm",
                    "class",
                    "start_sample",
                    "end_sample",
                    "start_sec",
                    "end_sec",
                    "duration_sec",
                    "num_beats",
                    "num_normal_beats",
                    "num_abnormal_beats",
                    "num_frames",
                ]
            )
        return

    fieldnames = list(segments[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segments)


def save_frames(frame_dir: Path, frames: np.ndarray) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame_path = frame_dir / f"frame_{index:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def export_record_frames(
    record_name: str,
    data_dir: Path,
    output_root: Path,
    config: ExportConfig,
) -> dict:
    record_dir = record_output_dir(output_root, record_name)
    if record_export_complete(output_root, record_name) and not config.overwrite:
        return {
            "record_name": record_name,
            "status": "skipped_existing",
            "record_dir": str(record_dir),
            "num_segments": None,
        }

    if config.overwrite and record_dir.exists():
        shutil.rmtree(record_dir)

    record, annotation = load_record_bundle(data_dir, record_name, config.annotation_extension)
    if config.lead < 0 or config.lead >= record.p_signal.shape[1]:
        raise ValueError(
            f"Lead index {config.lead} is out of bounds for record {record_name} with {record.p_signal.shape[1]} leads."
        )

    segments = build_segments(record_name, record, annotation, config)
    record_dir.mkdir(parents=True, exist_ok=True)

    exported_segments: list[dict] = []
    for segment in segments:
        frames, segment_meta = segment_to_frames(record, segment, config)
        segment_dir = record_dir / f"segment_{segment['segment_id']:04d}"
        frames_dir = segment_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        save_frames(frames_dir, frames)
        segment_meta["frames_dir"] = str(frames_dir.relative_to(output_root))
        write_json(segment_dir / "metadata.json", segment_meta)
        exported_segments.append(segment_meta)

    write_segments_csv(record_dir / "segments.csv", exported_segments)
    export_summary = {
        "record_name": record_name,
        "status": "exported",
        "num_segments": len(exported_segments),
        "record_dir": str(record_dir),
        "config": asdict(config),
    }
    write_json(record_dir / "export_summary.json", export_summary)
    return export_summary
