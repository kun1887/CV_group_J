from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_frame_pipeline import ExportConfig, ensure_records_available, export_record_frames, resolve_record_names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export MIT-BIH rhythm segments as ECG frame sequences for V-JEPA experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"), help="Directory containing WFDB records.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mitdb_vjepa_frames"),
        help="Directory where rendered frames and metadata will be saved.",
    )
    parser.add_argument("--db-name", default="mitdb", help="WFDB database name.")
    parser.add_argument(
        "--records",
        nargs="+",
        help="Explicit record names to process, e.g. --records 100 101 200.",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        help="Process only the first N records from the dataset record list.",
    )
    parser.add_argument("--annotation-extension", default="atr", help="WFDB annotation extension.")
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
        "--max-duration-sec",
        type=float,
        help="Optional cap on segment duration before framing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export records even if an existing completed export is present.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    record_names = resolve_record_names(
        db_name=args.db_name,
        selected_records=args.records,
        first_n=args.first_n,
    )
    ensure_records_available(
        db_name=args.db_name,
        data_dir=args.data_dir,
        record_names=record_names,
        annotation_extension=args.annotation_extension,
    )

    config = ExportConfig(
        db_name=args.db_name,
        annotation_extension=args.annotation_extension,
        lead=args.lead,
        seconds_per_frame=args.seconds_per_frame,
        image_size=args.image_size,
        pad=args.pad,
        line_thickness=args.line_thickness,
        draw_grid=not args.no_grid,
        grid_step=args.grid_step,
        max_duration_sec=args.max_duration_sec,
        overwrite=args.overwrite,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for record_name in record_names:
        summary = export_record_frames(
            record_name=record_name,
            data_dir=args.data_dir,
            output_root=args.output_dir,
            config=config,
        )
        summaries.append(summary)
        status = summary["status"]
        num_segments = summary["num_segments"]
        print(f"{record_name}: {status}" + (f" ({num_segments} segments)" if num_segments is not None else ""))

    run_summary = {
        "db_name": args.db_name,
        "records_requested": record_names,
        "num_records_requested": len(record_names),
        "num_exported": sum(1 for summary in summaries if summary["status"] == "exported"),
        "num_skipped_existing": sum(1 for summary in summaries if summary["status"] == "skipped_existing"),
        "output_dir": str(args.output_dir),
        "summaries": summaries,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(f"Run summary written to {args.output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
