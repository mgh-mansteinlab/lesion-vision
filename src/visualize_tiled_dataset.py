#!/usr/bin/env python
"""Dataset QA CLI moved to scripts/qa/visualize_tiled_dataset.py."""

import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

runpy.run_path(
    os.path.join(ROOT, "scripts", "qa", "visualize_tiled_dataset.py"),
    run_name="__main__",
)
