"""launch_experiment builds argv without running training."""
import importlib.util
from pathlib import Path

import yaml


def _load_launch_module(root: Path):
    path = root / "scripts" / "launch_experiment.py"
    spec = importlib.util.spec_from_file_location("launch_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_build_argv_from_baseline_yaml():
    root = Path(__file__).resolve().parents[1]
    build_argv = _load_launch_module(root).build_argv
    yml = root / "configs" / "experiments" / "baseline_multi_scale.yaml"
    cfg = yaml.safe_load(yml.read_text())
    assert cfg["train"]["tile_size"] == [448, 768]

    argv = build_argv(yml, root)
    assert "src/train.py" in argv[1] or argv[1].endswith("train.py")
    assert "--tile_size" in argv
    assert "448" in argv and "768" in argv
