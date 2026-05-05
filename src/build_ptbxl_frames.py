from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import wfdb

from ecg_frame_pipeline import (
    ExportConfig,
    record_export_complete,
    record_output_dir,
    save_frames,
    segment_to_frames,
    write_json,
    write_segments_csv,
)


PTBXL_DB_NAME = "ptb-xl"
PTBXL_METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download PTB-XL 100 Hz records and export each ECG recording as one frame sequence for V-JEPA experiments."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/data/ptbxl"),
        help="Directory containing PTB-XL metadata and downloaded WFDB files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/data/ptbxl_vjepa_frames"),
        help="Directory where rendered frames and metadata will be saved.",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Optional subset of PTB-XL ecg_id values to export, e.g. --records 1 2 3.",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        help="Export only the first N PTB-XL records from the metadata table.",
    )
    parser.add_argument("--lead", type=int, default=0, help="Signal lead index to render.")
    parser.add_argument(
        "--seconds-per-frame",
        type=float,
        default=1.0,
        help="How many ECG seconds each rendered frame should cover.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Square frame size in pixels.")
    parser.add_argument("--pad", type=int, default=18, help="Vertical padding in pixels.")
    parser.add_argument("--line-thickness", type=int, default=2, help="ECG trace thickness in pixels.")
    parser.add_argument("--grid-step", type=int, default=28, help="Grid spacing in pixels.")
    parser.add_argument("--no-grid", action="store_true", help="Disable ECG background grid.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export records even if an existing completed export is present.",
    )
    return parser


def ensure_ptbxl_metadata(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    missing_files = [name for name in PTBXL_METADATA_FILES if not (data_dir / name).exists()]
    if not missing_files:
        print(f"PTB-XL metadata already available in {data_dir}.")
        return

    print(f"Downloading PTB-XL metadata into {data_dir}: {', '.join(missing_files)}")
    wfdb.dl_files(PTBXL_DB_NAME, dl_dir=str(data_dir), files=missing_files, keep_subdirs=False)


def load_ptbxl_rows(data_dir: Path) -> list[dict]:
    metadata_path = data_dir / "ptbxl_database.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing PTB-XL metadata file: {metadata_path}")

    rows: list[dict] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["ecg_id"] = str(row["ecg_id"])
            row["scp_codes_dict"] = ast.literal_eval(row["scp_codes"])
            rows.append(row)
    return rows


def select_ptbxl_rows(rows: list[dict], selected_records: set[str] | None, first_n: int | None) -> list[dict]:
    if selected_records:
        filtered = [row for row in rows if row["ecg_id"] in selected_records]
        missing = sorted(selected_records - {row["ecg_id"] for row in filtered})
        if missing:
            raise ValueError(f"Requested PTB-XL ecg_id values not found in metadata: {', '.join(missing)}")
        return filtered

    if first_n is not None:
        if first_n <= 0:
            raise ValueError("--first-n must be a positive integer.")
        return rows[:first_n]

    return rows


def maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def record_files_exist(data_dir: Path, filename_lr: str) -> bool:
    return (data_dir / f"{filename_lr}.dat").exists() and (data_dir / f"{filename_lr}.hea").exists()


def ensure_ptbxl_records_available(data_dir: Path, rows: list[dict]) -> None:
    files_to_download: list[str] = []
    missing_record_ids: list[str] = []
    for row in rows:
        filename_lr = row["filename_lr"]
        if record_files_exist(data_dir, filename_lr):
            continue
        files_to_download.extend([f"{filename_lr}.dat", f"{filename_lr}.hea"])
        missing_record_ids.append(row["ecg_id"])

    if not files_to_download:
        print(f"All {len(rows)} requested PTB-XL 100 Hz record(s) are already available in {data_dir}.")
        return

    print(
        f"Downloading {len(missing_record_ids)} missing PTB-XL 100 Hz record(s) into {data_dir}: "
        f"{', '.join(missing_record_ids)}"
    )
    wfdb.dl_files(PTBXL_DB_NAME, dl_dir=str(data_dir), files=files_to_download, keep_subdirs=True)
    print("PTB-XL record download phase finished.")


def classify_ptbxl_record(scp_codes: dict[str, float]) -> str:
    return "normal rhythm" if "NORM" in scp_codes else "not-normal rhythm"


def export_ptbxl_record(
    row: dict,
    data_dir: Path,
    output_root: Path,
    config: ExportConfig,
) -> dict:
    record_name = row["ecg_id"]
    record_dir = record_output_dir(output_root, record_name)
    if record_export_complete(output_root, record_name) and not config.overwrite:
        print(f"Skipping PTB-XL record {record_name}: existing export found at {record_dir}.")
        return {
            "record_name": record_name,
            "status": "skipped_existing",
            "record_dir": str(record_dir),
            "num_segments": None,
        }

    if config.overwrite and record_dir.exists():
        print(f"Overwriting existing PTB-XL export for record {record_name} at {record_dir}.")
        shutil.rmtree(record_dir)

    record_path = str(data_dir / row["filename_lr"])
    print(f"Loading PTB-XL record {record_name} from {record_path}.")
    record = wfdb.rdrecord(record_path)
    if config.lead < 0 or config.lead >= record.p_signal.shape[1]:
        raise ValueError(
            f"Lead index {config.lead} is out of bounds for PTB-XL record {record_name} "
            f"with {record.p_signal.shape[1]} leads."
        )

    scp_codes_dict = row["scp_codes_dict"]
    sorted_scp_codes = sorted(scp_codes_dict)
    signal = record.p_signal
    fs = float(record.fs)
    segment = {
        "segment_id": 0,
        "record_name": record_name,
        "lead_index": int(config.lead),
        "lead_name": record.sig_name[config.lead],
        "rhythm": ",".join(sorted_scp_codes),
        "class": classify_ptbxl_record(scp_codes_dict),
        "start_sample": 0,
        "end_sample": int(signal.shape[0]),
        "start_sec": 0.0,
        "end_sec": float(signal.shape[0] / fs),
        "duration_sec": float(signal.shape[0] / fs),
        "num_beats": None,
        "num_normal_beats": None,
        "num_abnormal_beats": None,
    }

    print(
        f"PTB-XL record {record_name}: exporting single segment "
        f"(class={segment['class']}, duration={segment['duration_sec']:.2f}s, labels={segment['rhythm']})."
    )
    frames, segment_meta = segment_to_frames(record, segment, config)
    segment_meta.update(
        {
            "ecg_id": int(record_name),
            "patient_id": maybe_int(row.get("patient_id")),
            "filename_lr": row["filename_lr"],
            "filename_hr": row.get("filename_hr"),
            "scp_codes": sorted_scp_codes,
            "scp_codes_with_likelihood": scp_codes_dict,
            "report": row.get("report"),
            "strat_fold": maybe_int(row.get("strat_fold")),
        }
    )

    record_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = record_dir / "segment_0000"
    frames_dir = segment_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    save_frames(frames_dir, frames)
    segment_meta["frames_dir"] = str(frames_dir.relative_to(output_root))
    write_json(segment_dir / "metadata.json", segment_meta)
    write_segments_csv(record_dir / "segments.csv", [segment_meta])

    export_summary = {
        "record_name": record_name,
        "status": "exported",
        "num_segments": 1,
        "record_dir": str(record_dir),
        "config": asdict(config),
        "sampling_rate_hz": int(record.fs),
        "source_dataset": PTBXL_DB_NAME,
    }
    write_json(record_dir / "export_summary.json", export_summary)
    print(f"Finished exporting PTB-XL record {record_name} to {record_dir}.")
    return export_summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_ptbxl_metadata(args.data_dir)
    rows = load_ptbxl_rows(args.data_dir)
    selected_rows = select_ptbxl_rows(rows, set(args.records) if args.records else None, args.first_n)
    ensure_ptbxl_records_available(args.data_dir, selected_rows)

    config = ExportConfig(
        db_name=PTBXL_DB_NAME,
        annotation_extension="",
        lead=args.lead,
        seconds_per_frame=args.seconds_per_frame,
        image_size=args.image_size,
        pad=args.pad,
        line_thickness=args.line_thickness,
        draw_grid=not args.no_grid,
        grid_step=args.grid_step,
        max_duration_sec=None,
        overwrite=args.overwrite,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for row in selected_rows:
        summary = export_ptbxl_record(
            row=row,
            data_dir=args.data_dir,
            output_root=args.output_dir,
            config=config,
        )
        summaries.append(summary)
        status = summary["status"]
        num_segments = summary["num_segments"]
        print(f"{row['ecg_id']}: {status}" + (f" ({num_segments} segments)" if num_segments is not None else ""))

    run_summary = {
        "db_name": PTBXL_DB_NAME,
        "sampling_rate_hz": 100,
        "records_requested": [row["ecg_id"] for row in selected_rows],
        "num_records_requested": len(selected_rows),
        "num_exported": sum(1 for summary in summaries if summary["status"] == "exported"),
        "num_skipped_existing": sum(1 for summary in summaries if summary["status"] == "skipped_existing"),
        "output_dir": str(args.output_dir),
        "summaries": summaries,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(f"Run summary written to {args.output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
