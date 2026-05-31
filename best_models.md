# Best Models Parameters by Architecture
## Transformer on raw ecg

Here's the best run from `ptbxl-transformer-auc-sweep`:

**Best run:** `run-20260519_152650-yxdmrauy`

| Metric | Value |
|---|---|
| **best_val_roc_auc** | **0.8566** |
| test_roc_auc | 0.8504 |
| test_f1_tuned | 0.7872 |

**Hyperparameters:**

| Parameter | Value |
|---|---|
| batch_size | 128 |
| dropout | 0.3 |
| embed_dim | 128 |
| lr | 0.000362 |
| num_heads | 4 |
| num_layers | 4 |


## MLP on V-JEPA embeddings

**Best run:** `run-20260518_150333-5xy1jv5s`

| Metric | Value |
|---|---|
| **best_val_roc_auc** | **0.8676** |
| test_roc_auc | 0.8667 |
| test_f1_tuned | 0.7992 |

**Hyperparameters:**

| Parameter | Value |
|---|---|
| architecture | `deep_narrow` → `[512, 256, 128, 64]` |
| batch_size | 64 |
| dropout | 0.0 |
| lr | 0.00511 |



## LSTM on V-JEPA embeddings

**Best run:** `7x6xq4al`

| Metric | Value |
| --- | --- |
| **val_roc_auc** | **0.8563** |
| val_f1 | 0.8116 |
| val_recall | 0.8730 |
| val_precision | 0.7583 |
| threshold | 0.2828 |

**Hyperparameters:**

| Parameter | Value |
| --- | --- |
| hidden_size | 256 |
| num_layers | 4 |
| dropout | 0.0 |
| learning_rate | 0.000107 |
| weight_decay | 0.0001 |
| train_batch_size | 128 |
| epochs_requested | 60 |
| epochs_ran | 15 |
| early_stopping_metric | `val_roc_auc` |
| best_epoch | 5 |
