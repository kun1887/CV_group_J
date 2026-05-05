from __future__ import annotations

import argparse
import json
from pathlib import Path

from vjepa_embedding_utils import ensure_clip_embedding_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved ECG V-JEPA frame sequences into cached V-JEPA clip embeddings."
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
        help="Output cache directory. One .npz file is written per record unless a legacy .npz path is provided.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("src/data/vjepa_clip_embedding_experiments/embedding_summary.json"),
        help="Where to save a JSON summary of the generated cache.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/vjepa2-vitl-fpc64-256",
        help="Hugging Face V-JEPA model name.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used for embedding extraction.",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Optional subset of exported records to embed, e.g. --records 100 102 104.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Optional limit on the number of segments to embed.",
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
        "--batch-size",
        type=int,
        default=2,
        help="Number of clips per V-JEPA forward pass.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore an existing cache and rebuild embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_records = set(args.records) if args.records else None
    examples, clip_embeddings, meta = ensure_clip_embedding_cache(
        dataset_root=args.dataset_root,
        embedding_cache=args.embedding_cache,
        selected_records=selected_records,
        max_segments=args.max_segments,
        model_name=args.model_name,
        device_arg=args.device,
        batch_size=args.batch_size,
        target_num_frames=args.target_num_frames,
        clip_stride=args.clip_stride,
        force_recompute=args.force_recompute,
    )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    segment_labels = meta["segment_labels"]
    summary = {
        "embedding_cache": str(args.embedding_cache),
        "num_segments": int(len(examples)),
        "num_records": int(len(set(meta["segment_record_names"].tolist()))),
        "num_clips": int(len(meta["clip_indices"])),
        "clip_embedding_shape": list(clip_embeddings.shape),
        "segment_label_distribution": {
            "normal_rhythm": int((segment_labels == 0).sum()),
            "not_normal_rhythm": int((segment_labels == 1).sum()),
        },
        "records": sorted(set(meta["segment_record_names"].tolist())),
        "model_name": args.model_name,
        "target_num_frames": int(args.target_num_frames),
        "clip_stride": int(args.clip_stride),
        "cache_mode": "single_file" if args.embedding_cache.suffix == ".npz" else "per_record",
        "stored_representation": "per_clip_embeddings",
    }
    args.summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Clip embedding cache: {args.embedding_cache}")
    print(f"Summary: {args.summary_path}")
    print(f"Segments embedded: {len(examples)}")
    print(f"Clips embedded: {len(meta['clip_indices'])}")
    print(f"Clip embedding shape: {clip_embeddings.shape}")


if __name__ == "__main__":
    main()
