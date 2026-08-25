# Experiment log

Append one row per training or evaluation run that feeds the paper.

| Date | Experiment ID | Config / command | Checkpoint | Notes (metrics, figure/table) |
|------|---------------|------------------|------------|-------------------------------|
| YYYY-MM-DD | baseline_multi_scale | `scripts/run_experiment_train.sh configs/experiments/baseline_multi_scale.yaml` | `models/baseline_multi_scale/best_model.pth` | Paper Table 1 baseline |

**Batch runs:** `nohup bash scripts/run_all_experiments_nohup.sh …` appends machine-readable rows to [`experiments/EXPERIMENT_LOG_APPEND.tsv`](EXPERIMENT_LOG_APPEND.tsv) and detailed logs under [`experiments/logs/`](logs/). See [`experiments/README.md`](README.md).
