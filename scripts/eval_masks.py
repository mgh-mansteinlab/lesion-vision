#!/usr/bin/env python3
"""Compare a predicted RGB mask against ground truth."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.cli import main

if __name__ == "__main__":
    main()
