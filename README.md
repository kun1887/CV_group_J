# PTB-XL V-JEPA2 ECG Pipeline

This repository contains a binary PTB-XL classification pipeline that starts from one raw ECG lead, converts the signal into image frames, extracts V-JEPA2 embeddings, and trains downstream predictors. The binary target is `normal rhythm` versus `not-normal rhythm`, where `normal rhythm` is assigned when the PTB-XL SCP labels contain `NORM`.

## 1. Raw PTB-XL ECG to Frames

Responsible script: `src/build_ptbxl_frames.py`

This script downloads or reads PTB-XL 100 Hz WFDB records, selects one ECG lead, renders each 10-second recording as an ordered sequence of image frames, and writes per-record metadata.

Typical output:

```text
src/data/ptbxl_vjepa_frames/
  <record_id>/segment_0000/frame_0000.png
  <record_id>/segment_0000/metadata.json
```

Representative command:

```bash
python src/build_ptbxl_frames.py \
  --data-dir src/data/ptbxl \
  --output-dir src/data/ptbxl_vjepa_frames \
  --lead 0 \
  --seconds-per-frame 0.19 \
  --image-size 256
```

## 2. Frames to V-JEPA2 Embeddings

Responsible scripts:

```text
src/build_vjepa_clip_embeddings.py
src/build_vjepa_embeddings.py
```

Both scripts load the saved frame sequences, split each record into fixed-length V-JEPA clips, and run the V-JEPA2 encoder. For the fpc16 experiments, the model used is:

```text
facebook/vjepa2-vitl-fpc16-256-ssv2
```

`src/build_vjepa_clip_embeddings.py` saves the full per-clip sequence for each record. This cache is used by the LSTM and by the logistic-regression probe, which pools the clips internally.

```text
src/data/ptbxl_vjepa_clip_embedding_experiments_fpc16/records/<record_id>.npz
```

Representative command:

```bash
python src/build_vjepa_clip_embeddings.py \
  --dataset-root src/data/ptbxl_vjepa_frames \
  --model-name facebook/vjepa2-vitl-fpc16-256-ssv2 \
  --embedding-cache src/data/ptbxl_vjepa_clip_embedding_experiments_fpc16/records \
  --summary-path src/data/ptbxl_vjepa_clip_embedding_experiments_fpc16/embedding_summary.json
```

`src/build_vjepa_embeddings.py` pools V-JEPA clip embeddings into one embedding per record. This record-level cache is used by the MLP baseline.

Representative command:

```bash
python src/build_vjepa_embeddings.py \
  --dataset-root src/data/ptbxl_vjepa_frames \
  --model-name facebook/vjepa2-vitl-fpc16-256-ssv2 \
  --embedding-cache src/data/ptbxl_vjepa_embeddings_fpc16/records \
  --summary-path src/data/ptbxl_vjepa_embeddings_fpc16/embedding_summary.json
```

## 3. Logistic Regression on Pooled V-JEPA2 Embeddings

Responsible script: `src/linear_probe_pooled_clip_embeddings.py`

This script pools each record's cached clip embeddings, fits a standardized logistic regression probe, tunes an optional decision threshold on the validation split, and reports validation/test metrics using the PTB-XL stratified folds.

Representative command:

```bash
python src/linear_probe_pooled_clip_embeddings.py \
  --embedding-cache src/data/ptbxl_vjepa_clip_embedding_experiments_fpc16/records \
  --dataset-root src/data/ptbxl_vjepa_frames \
  --output-dir src/data/ptbxl_vjepa_linear_probe_fpc16
```

## 4. MLP on V-JEPA2 Embeddings

Responsible script: `src/ptbxl_mlp.py`

This script trains a multilayer perceptron on record-level V-JEPA embeddings using the PTB-XL train/validation/test folds.

Representative command:

```bash
python src/ptbxl_mlp.py \
  --embedding-cache src/data/ptbxl_vjepa_embeddings_fpc16/records \
  --metadata src/data/ptbxl/ptbxl_database.csv \
  --output-dir src/data/ptbxl_results/mlp
```

## 5. LSTM on V-JEPA2 Clip Sequences

Responsible script: `src/lstm_with_clip_embedding.py`

This script trains an LSTM over the sequence of cached V-JEPA clip embeddings for each record. It supports cached-only training, W&B sweeps, early stopping, and validation-based threshold tuning.

Representative command:

```bash
python src/lstm_with_clip_embedding.py \
  --dataset-root src/data/ptbxl_vjepa_frames \
  --embedding-cache src/data/ptbxl_vjepa_clip_embedding_experiments_fpc16/records \
  --output-dir src/data/ptbxl_vjepa_lstm_sweeps_fpc16 \
  --cached-only \
  --tune-threshold-for-f1
```

## 6. Transformer on Raw ECG

Responsible scripts:

```text
src/build_ecg_token_cache.py
src/ptbxl_transformer.py
```

`src/build_ecg_token_cache.py` converts raw PTB-XL ECG signals into fixed-length waveform tokens for one or more leads. `src/ptbxl_transformer.py` trains a transformer classifier directly on those raw ECG tokens.

Representative commands:

```bash
python src/build_ecg_token_cache.py \
  --data-dir src/data/ptbxl \
  --cache-root src/data/t_caches \
  --lead 0 \
  --window-sec 1.0

python src/ptbxl_transformer.py \
  --data-dir src/data/ptbxl \
  --cache-root src/data/t_caches \
  --leads 0 \
  --window-sec 1.0 \
  --output-dir src/data/ptbxl_results/transformer
```

## Split Convention

The supervised PTB-XL experiments use the official `strat_fold` split:

```text
train: folds 1-8
validation: fold 9
test: fold 10
```
