#!/usr/bin/env python
"""CLI wrapper. Implementation lives in src.data.tiling."""

import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

runpy.run_module("src.data.tiling", run_name="__main__")
