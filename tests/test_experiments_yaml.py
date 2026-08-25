"""Experiment configs are valid YAML."""
from pathlib import Path

import pytest
import yaml


def test_all_experiment_yamls_parse():
    root = Path(__file__).resolve().parents[1]
    exp_dir = root / "configs" / "experiments"
    assert exp_dir.is_dir(), f"Missing {exp_dir}"
    for p in sorted(exp_dir.glob("*.yaml")):
        data = yaml.safe_load(p.read_text())
        assert "train" in data or "protocol" in data, p.name
