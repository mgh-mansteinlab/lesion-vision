"""Multi-architecture benchmark + task-difficulty demonstration.

- ``run_benchmark``: standardized evaluation of every trained backbone (and,
  optionally, SAM3) on a fixed ground-truth set, reporting per-class IoU/Dice,
  means, lesion-only metrics, inference latency, and parameter counts.
- ``task_difficulty``: weak unsupervised/threshold baselines vs trained models,
  showing the task needs specialized supervised learning.
"""
