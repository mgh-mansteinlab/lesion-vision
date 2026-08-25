#!/usr/bin/env python
"""Training entrypoint. Equivalent to ``python -m src.train.cli``."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train.cli import main

if __name__ == "__main__":
    main()
