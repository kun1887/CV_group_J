# CV_group_J

Computer Vision 2025-2026 group project.

This repository contains a preprocessing and evaluation pipeline for:

1. reading MIT-BIH ECG records with `wfdb`
2. converting MIT-BIH rhythm segments or PTB-XL recordings into ECG image frames suitable for V-JEPA-style video input
3. converting saved frame sequences into cached V-JEPA pooled segment embeddings
4. converting saved frame sequences into cached V-JEPA clip embeddings
5. evaluating pooled segment embeddings with unsupervised clustering and supervised baselines

## Pipeline Overview

The project is organized as a staged pipeline. Each stage writes files to disk so later stages can be rerun without recomputing earlier work.

```text
MIT-BIH WFDB records
-> rhythm segments
-> ECG frame sequences on disk
-> V-JEPA pooled segment embeddings cache
-> clustering / linear probe / SVM experiments

or

PTB-XL 100 Hz WFDB records
-> one 10-second ECG recording treated as one segment
-> ECG frame sequences on disk
-> V-JEPA pooled segment embeddings cache
-> downstream experiments

or

MIT-BIH WFDB records
-> rhythm segments
-> ECG frame sequences on disk
-> V-JEPA clip embeddings cache
-> temporal sequence models (next step)
```

There are four main execution stages:

1. `build_ecg_frames.py`
   - downloads MIT-BIH records if needed
   - extracts rhythm segments from WFDB annotations
   - renders each segment into a sequence of ECG frames

2. `build_ptbxl_frames.py`
   - downloads PTB-XL metadata and 100 Hz WFDB files
   - treats each 10-second PTB-XL recording as one segment
   - renders each recording into a sequence of ECG frames

3. `build_vjepa_embeddings.py`
   - reads saved segment frame folders
   - runs V-JEPA on fixed-length clips sampled from each segment
   - pools clip embeddings into one embedding per segment
   - saves the embeddings locally in a cache file

4. `build_vjepa_clip_embeddings.py`
   - reads saved segment frame folders
   - runs V-JEPA on fixed-length clips sampled from each segment
   - saves one embedding per clip, with segment-aligned metadata

5. analysis scripts
   - `clustering_with_embedding.py`
   - `linear_probe_with_embedding.py`
   - `svm_with_embedding.py`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `torchvision` is required because `AutoVideoProcessor` for V-JEPA depends on it.
- `transformers` downloads the selected V-JEPA checkpoint from Hugging Face on first use.
- The `urllib3` / `LibreSSL` warning on macOS is noisy but not the root cause of the project logic.

## Scripts

### 1. Build ECG Frames

Script:

- [src/build_ecg_frames.py](/Users/kunzhan/github/kun1887/CV_group_J/src/build_ecg_frames.py)

Core logic:

- [src/ecg_frame_pipeline.py](/Users/kunzhan/github/kun1887/CV_group_J/src/ecg_frame_pipeline.py)

What it does:

- resolves which MIT-BIH records to process
- downloads missing WFDB files
- reads rhythm annotations from `aux_note`
- defines one segment per rhythm interval
- labels `(N)` as `normal rhythm` and everything else as `not-normal rhythm`
- renders each segment into constant-time ECG frames
- skips records that already have a completed export unless `--overwrite` is used

Recommended command:

```bash
python3 src/build_ecg_frames.py \
  --data-dir src/data/mitdb \
  --output-dir src/data/mitdb_vjepa_frames
```

Record selection options:

- all records:

```bash
python3 src/build_ecg_frames.py \
  --data-dir src/data/mitdb \
  --output-dir src/data/mitdb_vjepa_frames
```

- only specific records:

```bash
python3 src/build_ecg_frames.py \
  --data-dir src/data/mitdb \
  --output-dir src/data/mitdb_vjepa_frames \
  --records 100 102 104
```

- only the first `n` records:

```bash
python3 src/build_ecg_frames.py \
  --data-dir src/data/mitdb \
  --output-dir src/data/mitdb_vjepa_frames \
  --first-n 5
```

Important note:

- `build_ecg_frames.py` defaults to `data/mitdb_vjepa_frames`
- the downstream scripts default to `src/data/mitdb_vjepa_frames`

To keep the whole pipeline aligned, use:

```bash
--output-dir src/data/mitdb_vjepa_frames
```

unless you also override the downstream `--dataset-root`.

### 2. Build PTB-XL Frames

Script:

- [src/build_ptbxl_frames.py](/Users/kunzhan/github/kun1887/CV_group_J/src/build_ptbxl_frames.py)

Core rendering logic reused from:

- [src/ecg_frame_pipeline.py](/Users/kunzhan/github/kun1887/CV_group_J/src/ecg_frame_pipeline.py)

What it does:

- downloads PTB-XL metadata files
- downloads the 100 Hz WFDB waveform files referenced by `filename_lr`
- treats each PTB-XL ECG recording as one segment
- renders each 10-second recording into ECG frames
- saves per-record metadata in the same general frame-export structure used by MIT-BIH

Recommended command:

```bash
python3 src/build_ptbxl_frames.py
```

By default it writes:

- raw PTB-XL files:
  `src/data/ptbxl`
- rendered frame dataset:
  `src/data/ptbxl_vjepa_frames`

Examples:

- export only the first `n` PTB-XL records:

```bash
python3 src/build_ptbxl_frames.py --first-n 100
```

- export only selected PTB-XL `ecg_id` values:

```bash
python3 src/build_ptbxl_frames.py --records 1 2 3
```

Important note:

- PTB-XL does not use MIT-BIH-style rhythm intervals in `aux_note`
- each 10-second PTB-XL recording is treated as a single segment/sample
- the downstream embedding scripts can be reused by pointing `--dataset-root` to `src/data/ptbxl_vjepa_frames`

### 2bis. Multi-label Transformer baseline on raw PTB-XL ECG

Script:

- [src/transformer_ptbxl_multilabel.py](src/transformer_ptbxl_multilabel.py)

What it does:

- downloads PTB-XL metadata and 100 Hz WFDB files if missing
- parses `scp_codes` and maps them to the five PTB-XL diagnostic super-classes
  (`NORM`, `MI`, `STTC`, `CD`, `HYP`) using `scp_statements.csv`
- skips records that have no diagnostic super-class label
- splits records by the official `strat_fold` column:
  - train: folds 1..8
  - val: fold 9 (used for per-class threshold tuning)
  - test: fold 10 (final reported metrics)
- trains a Transformer encoder on raw lead-0 ECG (1s tokens, 256-d embedding,
  4 layers, 8 heads, FFN=1024, CLS token, sinusoidal positional encoding) with
  a multi-label head of 5 logits, `BCEWithLogitsLoss` and class-balanced
  `pos_weight`, AdamW + cosine LR, gradient clipping, early stopping on val
  macro-AUC
- tunes per-class thresholds on fold 9 to maximize F1, then reports macro-AUC,
  macro-F1, and per-class AUC/AP/precision/recall/F1/confusion matrix on fold 10
- saves `summary.json` and `model.pt`

Recommended command:

```bash
python3 src/transformer_ptbxl_multilabel.py
```

Quick smoke test on a small subset:

```bash
python3 src/transformer_ptbxl_multilabel.py --max-records 200 --epochs 3
```

### 3. Build V-JEPA Embeddings

Script:

- [src/build_vjepa_embeddings.py](/Users/kunzhan/github/kun1887/CV_group_J/src/build_vjepa_embeddings.py)

Shared embedding utilities:

- [src/vjepa_embedding_utils.py](/Users/kunzhan/github/kun1887/CV_group_J/src/vjepa_embedding_utils.py)

What it does:

- reads exported frame sequences from the dataset root
- splits long segments into overlapping fixed-length clips
- runs the V-JEPA encoder on those clips
- averages clip embeddings to obtain one embedding per segment
- saves the result as a local `.npz` cache

Important note:

- V-JEPA computes one embedding per clip internally
- this script then averages those clip embeddings
- the saved cache contains one final embedding per segment, not one embedding per clip

Recommended command:

```bash
python3 src/build_vjepa_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames
```

By default it writes:

- embedding cache directory:
  [src/data/vjepa_embedding_experiments/records](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_embedding_experiments/records)
- embedding summary:
  [src/data/vjepa_embedding_experiments/embedding_summary.json](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_embedding_experiments/embedding_summary.json)

Examples:

- embed only selected records:

```bash
python3 src/build_vjepa_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --records 100 102 104
```

- force recomputation of the cache:

```bash
python3 src/build_vjepa_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --force-recompute
```

### 4. Build V-JEPA Clip Embeddings

Script:

- [src/build_vjepa_clip_embeddings.py](/Users/kunzhan/github/kun1887/CV_group_J/src/build_vjepa_clip_embeddings.py)

Shared embedding utilities:

- [src/vjepa_embedding_utils.py](/Users/kunzhan/github/kun1887/CV_group_J/src/vjepa_embedding_utils.py)

What it does:

- reads exported frame sequences from the dataset root
- splits long segments into overlapping fixed-length clips
- runs the V-JEPA encoder on those clips
- saves one embedding per clip instead of pooling them to segment level
- saves clip-level metadata so clips can be regrouped into ordered segment sequences later

This stage is intended for temporal models such as:

- LSTM
- GRU
- Transformer

Recommended command:

```bash
python3 src/build_vjepa_clip_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames
```

By default it writes:

- clip embedding cache directory:
  [src/data/vjepa_clip_embedding_experiments/records](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_clip_embedding_experiments/records)
- clip embedding summary:
  [src/data/vjepa_clip_embedding_experiments/embedding_summary.json](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_clip_embedding_experiments/embedding_summary.json)

Examples:

- embed only selected records:

```bash
python3 src/build_vjepa_clip_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --records 100 102 104
```

- force recomputation of the cache:

```bash
python3 src/build_vjepa_clip_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --force-recompute
```

Each saved clip embedding row has aligned metadata fields such as:

- `clip_record_names`
- `clip_segment_ids`
- `clip_labels`
- `clip_label_text`
- `clip_rhythms`
- `clip_indices`
- `clip_start_frames`
- `clip_end_frames`

The clip label is inherited from the parent segment. That is correct for the current dataset, because supervision is available at the segment level, not at the individual clip level.

### 5. Clustering on Embeddings

Script:

- [src/clustering_with_embedding.py](/Users/kunzhan/github/kun1887/CV_group_J/src/clustering_with_embedding.py)

What it does:

- loads the saved embedding cache
- if needed, recomputes the embedding cache through `vjepa_embedding_utils`
- standardizes embeddings
- reduces them with PCA for clustering
- compares multiple clustering methods against the known rhythm labels

Methods currently evaluated:

- `kmeans`
- `gaussian_mixture`
- `agglomerative_ward`
- `agglomerative_average`
- `spectral`
- `birch`
- `dbscan`

Recommended command:

```bash
python3 src/clustering_with_embedding.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --embedding-cache src/data/vjepa_embedding_experiments/records
```

Outputs:

- [clustering_results.csv](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_embedding_experiments/clustering_results.csv)
- PCA plots for true labels and each clustering method

### 6. Linear Probe on Embeddings

Script:

- [src/linear_probe_with_embedding.py](/Users/kunzhan/github/kun1887/CV_group_J/src/linear_probe_with_embedding.py)

What it does:

- loads the saved embedding cache
- if needed, recomputes the cache through `vjepa_embedding_utils`
- trains a linear classifier directly on raw V-JEPA embeddings
- reports both segment-level stratified CV and leave-one-record-out evaluation

Recommended command:

```bash
python3 src/linear_probe_with_embedding.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --embedding-cache src/data/vjepa_embedding_experiments/records
```

Outputs:

- [linear_probe_summary.json](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_linear_probe/linear_probe_summary.json)
- `leave_one_record_out_results.csv`
- `true_labels_pca.png`

### 7. SVM on Embeddings

Script:

- [src/svm_with_embedding.py](/Users/kunzhan/github/kun1887/CV_group_J/src/svm_with_embedding.py)

What it does:

- loads the saved pooled segment embedding cache
- if needed, recomputes the cache through `vjepa_embedding_utils`
- trains an SVM on raw V-JEPA embeddings
- reports both stratified CV and leave-one-record-out evaluation

Recommended command:

```bash
python3 src/svm_with_embedding.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --embedding-cache src/data/vjepa_embedding_experiments/records
```

Outputs:

- [svm_summary.json](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_svm/svm_summary.json)
- `leave_one_record_out_results.csv`
- `true_labels_pca.png`

### 8. Tiny LSTM on Clip Embeddings

Script:

- [src/tiny_lstm_with_clip_embedding.py](/Users/kunzhan/github/kun1887/CV_group_J/src/tiny_lstm_with_clip_embedding.py)

What it does:

- loads saved clip embeddings
- rebuilds ordered clip-embedding sequences per segment
- trains a small LSTM segment classifier
- saves validation and test predictions with `score_not_normal`

Split behavior:

- for MIT-BIH-style exports:
  - uses one held-out record for test
  - uses a random stratified validation split from the remaining records
- for PTB-XL exports:
  - automatically uses the native `strat_fold` split
  - folds `1-8` train
  - fold `9` validation
  - fold `10` test
  - `--held-out-record` and `--val-fraction` are ignored in this mode

Example PTB-XL run:

```bash
python3 src/tiny_lstm_with_clip_embedding.py \
  --dataset-root src/data/ptbxl_vjepa_frames \
  --embedding-cache src/data/ptbxl_vjepa_clip_embedding_experiments/records \
  --output-dir src/data/ptbxl_vjepa_tiny_lstm
```

## Dataset Layout

The saved frame dataset is organized by record, then by segment.

Example:

```text
src/data/mitdb_vjepa_frames/
  run_summary.json
  100/
    export_summary.json
    segments.csv
    segment_0000/
      metadata.json
      frames/
        frame_0000.png
        frame_0001.png
        ...
  102/
    export_summary.json
    segments.csv
    segment_0000/
      metadata.json
      frames/
        frame_0000.png
        ...
```

Meaning:

- one directory per record
- one directory per rhythm segment
- one PNG sequence per segment

PTB-XL exports follow the same directory shape, but each record contributes exactly one segment because each 10-second measurement is treated as one sample.

Example:

```text
src/data/ptbxl_vjepa_frames/
  run_summary.json
  1/
    export_summary.json
    segments.csv
    segment_0000/
      metadata.json
      frames/
        frame_0000.png
        ...
```

## Embedding Cache Layout

There are now two embedding cache families.

### Pooled Segment Embeddings

Default location:

- [src/data/vjepa_embedding_experiments/records](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_embedding_experiments/records)

Layout:

```text
src/data/vjepa_embedding_experiments/records/
  100.npz
  101.npz
  102.npz
  ...
```

Each record cache stores one row per segment:

- `embeddings`
- `record_names`
- `segment_ids`
- `labels`
- `label_text`
- `rhythms`

### Clip Embeddings

Default location:

- [src/data/vjepa_clip_embedding_experiments/records](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_clip_embedding_experiments/records)

Layout:

```text
src/data/vjepa_clip_embedding_experiments/records/
  100.npz
  101.npz
  102.npz
  ...
```

Each record cache stores one row per clip in `embeddings`, together with clip-level and segment-level metadata.

Clip-level fields:

- `clip_record_names`
- `clip_segment_ids`
- `clip_labels`
- `clip_label_text`
- `clip_rhythms`
- `clip_indices`
- `clip_start_frames`
- `clip_end_frames`

Segment-level fields:

- `segment_record_names`
- `segment_ids`
- `segment_labels`
- `segment_label_text`
- `segment_rhythms`
- `segment_num_frames`
- `segment_num_clips`

Important metadata files:

- `run_summary.json`
  - summary of a frame-export run across all requested records

- `export_summary.json`
  - summary for one record

- `segments.csv`
  - one row per rhythm segment in that record

- `metadata.json`
  - full per-segment information:
    - rhythm label
    - class
    - timing
    - beat counts
    - number of rendered frames
    - frame directory

For PTB-XL exports, `metadata.json` also preserves source label information such as:

- `ecg_id`
- `patient_id`
- `filename_lr`
- `filename_hr`
- `scp_codes`
- `scp_codes_with_likelihood`
- `report`
- `strat_fold`

## Embedding Cache Layout

The V-JEPA embedding cache is now stored by record as compressed NumPy archives inside:

- [src/data/vjepa_embedding_experiments/records](/Users/kunzhan/github/kun1887/CV_group_J/src/data/vjepa_embedding_experiments/records)

Example:

```text
src/data/vjepa_embedding_experiments/records/
  100.npz
  101.npz
  102.npz
  ...
```

Each per-record `.npz` file contains:

- `embeddings`
  - array of shape `(num_segments_in_record, embedding_dim)`
- `record_names`
- `segment_ids`
- `labels`
- `label_text`
- `rhythms`

These embeddings are reused by both:

- `clustering_with_embedding.py`
- `linear_probe_with_embedding.py`

This is the reason the embedding stage is now separated from the downstream analyses. It also means already embedded records do not need to be recomputed when you add new records.

## End-to-End Example

If you want to run the full pipeline on the first 5 records:

```bash
python3 src/build_ecg_frames.py \
  --data-dir src/data/mitdb \
  --output-dir src/data/mitdb_vjepa_frames \
  --first-n 5
```

```bash
python3 src/build_vjepa_embeddings.py \
  --dataset-root src/data/mitdb_vjepa_frames
```

```bash
python3 src/clustering_with_embedding.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --embedding-cache src/data/vjepa_embedding_experiments/records
```

```bash
python3 src/linear_probe_with_embedding.py \
  --dataset-root src/data/mitdb_vjepa_frames \
  --embedding-cache src/data/vjepa_embedding_experiments/records
```

## Design Notes

### Rhythm Segments

Segments are derived from WFDB rhythm annotations in `aux_note`.

Default labeling:

- `(N)` -> `normal rhythm`
- everything else -> `not-normal rhythm`

This is a coarse binary rhythm label. It is segment-level, not frame-level and not beat-level.

### Frame Rendering

Each frame represents a fixed duration of ECG, controlled by `--seconds-per-frame`.

Default:

- `1.0` second per frame in `build_ecg_frames.py`

Rendering uses:

- constant time per frame
- global min/max scaling within a segment
- blank right-side padding for final partial frames

### V-JEPA Clip Construction

V-JEPA does not receive raw ECG samples directly. It receives image clips built from saved ECG frames.

For each segment:

1. load all PNG frames for the segment
2. split into fixed-length clips of `target_num_frames`
3. if the segment is short, resample to `target_num_frames`
4. if the segment is long, create multiple overlapping clips using `clip_stride`
5. run V-JEPA on each clip
6. average clip embeddings to get one segment embedding

Important consequence:

- V-JEPA produces one embedding per clip during computation
- the saved cache does not store clip-level embeddings
- the saved cache stores one final pooled embedding per segment

Default embedding settings:

- `target_num_frames = 16`
- `clip_stride = 8`

## Evaluation Notes

### Clustering

Clustering is unsupervised, but evaluation compares cluster assignments to the known labels.

Most relevant metrics:

- `ARI`
- `NMI`
- `purity`
- `silhouette`

### Linear Probe

The linear probe is supervised.

Two evaluation views are reported:

- stratified CV across segments
  - optimistic because segments from the same record can appear in both train and test

- leave-one-record-out
  - stronger estimate of generalization across recordings

For this project, leave-one-record-out is the more meaningful result.

## Common Pitfalls

### Downstream scripts cannot find frames

Cause:

- `build_ecg_frames.py` wrote to `data/mitdb_vjepa_frames`
- downstream scripts are still looking in `src/data/mitdb_vjepa_frames`

Fix:

- use `--output-dir src/data/mitdb_vjepa_frames` during frame export
- or override `--dataset-root` in later scripts

### V-JEPA preprocessing fails with `AutoVideoProcessor` import errors

Cause:

- `torchvision` is missing

Fix:

```bash
pip install torchvision
```

### Embeddings are recomputed unexpectedly

Cause:

- the selected records or segment subset no longer match the saved cache metadata

Behavior:

- the scripts detect this and rebuild the cache automatically

## File Reference

- [src/ecg_frame_pipeline.py](/Users/kunzhan/github/kun1887/CV_group_J/src/ecg_frame_pipeline.py)
  - frame-export logic

- [src/build_ecg_frames.py](/Users/kunzhan/github/kun1887/CV_group_J/src/build_ecg_frames.py)
  - CLI for WFDB -> frame export

- [src/vjepa_embedding_utils.py](/Users/kunzhan/github/kun1887/CV_group_J/src/vjepa_embedding_utils.py)
  - shared frame -> embedding cache logic

- [src/build_vjepa_embeddings.py](/Users/kunzhan/github/kun1887/CV_group_J/src/build_vjepa_embeddings.py)
  - CLI for building the embedding cache

- [src/clustering_with_embedding.py](/Users/kunzhan/github/kun1887/CV_group_J/src/clustering_with_embedding.py)
  - unsupervised clustering experiments

- [src/linear_probe_with_embedding.py](/Users/kunzhan/github/kun1887/CV_group_J/src/linear_probe_with_embedding.py)
  - supervised linear-probe evaluation
