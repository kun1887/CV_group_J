from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


FS = 100  # PTB-XL 100 Hz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache PTB-XL ECG signals as fixed-length tokens for Transformer training.")
    parser.add_argument("--data-dir", type=Path, default=Path("src/data/ptbxl"))
    parser.add_argument("--cache-root", type=Path, default=Path("src/data/t_caches"))
    parser.add_argument("--window-sec", type=float, default=1.0, help="Token window size in seconds (e.g. 1.0, 0.5, 0.1)")
    parser.add_argument("--lead", type=int, default=0, help="Lead index to use (0=I, 1=II, ...)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Fractional overlap between windows (0.0 = no overlap, 0.5 = 50%)")
    return parser.parse_args()


def load_metadata(data_dir: Path) -> list[dict]:
    rows = []
    with (data_dir / "ptbxl_database.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["ecg_id"] = str(int(float(row["ecg_id"])))
            row["strat_fold"] = int(float(row["strat_fold"]))
            row["scp_codes_dict"] = ast.literal_eval(row["scp_codes"])
            rows.append(row)
    return rows


def load_signal(data_dir: Path, filename_lr: str) -> np.ndarray | None:
    try:
        import wfdb
        rec = wfdb.rdrecord(str(data_dir / filename_lr))
        return rec.p_signal.astype(np.float32)  # (1000, 12)
    except Exception:
        return None


def signal_to_tokens(signal: np.ndarray, lead: int, window_samples: int, stride: int) -> np.ndarray:
    lead_signal = signal[:, lead]
    starts = list(range(0, len(lead_signal) - window_samples + 1, stride))
    # Mirror V-JEPA clip builder: always include a window anchored at the end of the signal
    last_start = len(lead_signal) - window_samples
    if starts[-1] != last_start:
        starts.append(last_start)
    tokens = np.stack([lead_signal[s:s + window_samples] for s in starts], axis=0)
    return tokens.astype(np.float32)


def main() -> None:
    args = parse_args()

    window_samples = int(args.window_sec * FS)
    stride = max(1, int(window_samples * (1.0 - args.overlap)))

    # Directory name encodes resolution, overlap (if any), and lead
    ov_tag = f"_ov{int(round(args.overlap * 100))}" if args.overlap > 0 else ""
    resolution_name = f"{args.window_sec:.2f}s{ov_tag}_lead{args.lead}"
    cache_dir = args.cache_root / resolution_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "cache.npz"

    print(f"Window: {args.window_sec}s ({window_samples} samples) | Stride: {stride} samples | Lead: {args.lead}")
    print(f"Cache path: {cache_path}")

    rows = load_metadata(args.data_dir)
    print(f"Found {len(rows)} records in metadata.")

    all_tokens, all_labels, all_folds, all_record_ids = [], [], [], []
    skipped = 0

    for i, row in enumerate(rows):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(rows)}...")
        signal = load_signal(args.data_dir, row["filename_lr"])
        if signal is None:
            skipped += 1
            continue
        tokens = signal_to_tokens(signal, args.lead, window_samples, stride)
        label = 0 if "NORM" in row["scp_codes_dict"] else 1
        all_tokens.append(tokens)
        all_labels.append(label)
        all_folds.append(row["strat_fold"])
        all_record_ids.append(row["ecg_id"])

    if skipped:
        print(f"Skipped {skipped} records (missing files).")

    tokens_array = np.stack(all_tokens, axis=0)  # (N, num_tokens, window_samples)
    labels_array = np.array(all_labels, dtype=np.int32)
    folds_array = np.array(all_folds, dtype=np.int32)
    record_ids_array = np.array(all_record_ids)

    print(f"Tokens shape: {tokens_array.shape}")
    print(f"Saving to {cache_path}...")
    np.savez_compressed(
        cache_path,
        tokens=tokens_array,
        labels=labels_array,
        folds=folds_array,
        record_ids=record_ids_array,
    )

    meta = {
        "window_sec": args.window_sec,
        "window_samples": window_samples,
        "stride": stride,
        "overlap": args.overlap,
        "lead": args.lead,
        "fs": FS,
        "num_records": int(len(all_tokens)),
        "num_tokens_per_record": int(tokens_array.shape[1]),
        "tokens_shape": list(tokens_array.shape),
        "skipped": skipped,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. {len(all_tokens)} records cached.")
    print(f"Tokens per record: {tokens_array.shape[1]}")
    print(f"Meta saved to {cache_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
