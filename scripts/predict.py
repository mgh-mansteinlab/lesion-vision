#!/usr/bin/env python
"""Batch prediction CLI (canonical). See scripts/predict_directory.py for the implementation."""

import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "predict_directory.py"),
    run_name="__main__",
)
