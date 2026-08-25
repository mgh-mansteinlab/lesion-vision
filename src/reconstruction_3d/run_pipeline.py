"""Driver for the LSA-3D registration + 3D surface reconstruction pipeline.

Usage:
    python run_pipeline.py --masks /path/to/masks
    python run_pipeline.py --masks /path/to/masks --samples /path/to/tissue_scans
    python run_pipeline.py --masks /path/to/masks --work /path/to/output 3 4 5 6
    python run_pipeline.py --list

Stages:
    1  classify masks -> labels + meta.json
    2  downsample NBTC tissue scans (skipped if --samples not given)
    3  pairwise registration -> aligned/
    4  lesion cleanup + column tracking -> aligned_clean/
    5  3D model + interactive HTML -> model/
    6  static PNG snapshots -> model/*.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(ROOT, "pipeline")

STAGES = {
    1: ("s1_extract.py", "classify masks -> labels + meta.json"),
    2: ("s2_tissue.py", "downsample NBTC tissue scans -> tissue/"),
    3: ("s3_register.py", "pairwise registration -> aligned/"),
    4: ("s4_clean.py", "lesion cleanup + column tracking -> aligned_clean/"),
    5: ("s5_build_model.py", "3D marching-cubes surfaces -> model/"),
    6: ("s6_snapshot.py", "static PNG snapshots -> model/*.png"),
}


def run_stage(num: int) -> None:
    script, desc = STAGES[num]
    print(f"\n=== stage {num}: {script}  ({desc}) ===", flush=True)
    subprocess.run([sys.executable, os.path.join(PIPE, script)], check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LSA-3D: serial-section registration and 3D lesion surface reconstruction",
    )
    parser.add_argument(
        "--masks",
        required=False,
        help="Path to directory of color-coded mask PNGs (green/blue/red labels)",
    )
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional path to original NBTC tissue-scan PNGs (same FOV as masks; improves registration)",
    )
    parser.add_argument(
        "--work",
        default=None,
        help="Output working directory (default: ./output under the project folder)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show stage map and exit",
    )
    parser.add_argument(
        "stages",
        nargs="*",
        type=int,
        help="Stage numbers to run (default: all applicable stages)",
    )
    return parser.parse_args(argv)


def resolve_stages(requested: list[int] | None, has_samples: bool) -> list[int]:
    if requested:
        stages = requested
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise SystemExit(f"unknown stage(s): {unknown}; valid stages are {sorted(STAGES)}")
    else:
        stages = sorted(STAGES)

    if 2 in stages and not has_samples:
        print("note: skipping stage 2 (no --samples path provided)")
        stages = [s for s in stages if s != 2]
    return stages


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.list:
        for num, (script, desc) in STAGES.items():
            print(f"{num}  {script:<20} {desc}")
        return

    if not args.masks:
        raise SystemExit("required: --masks /path/to/mask/pngs")

    masks = os.path.abspath(args.masks)
    if not os.path.isdir(masks):
        raise SystemExit(f"--masks is not a directory: {masks}")

    work = os.path.abspath(args.work or os.path.join(ROOT, "output"))
    os.makedirs(work, exist_ok=True)

    os.environ["LSA_SRC"] = masks
    os.environ["LSA_WORK"] = work

    samples = None
    if args.samples:
        samples = os.path.abspath(args.samples)
        if not os.path.isdir(samples):
            raise SystemExit(f"--samples is not a directory: {samples}")
        os.environ["LSA_TISSUE_SRC"] = samples
    else:
        os.environ.pop("LSA_TISSUE_SRC", None)

    print(f"masks (LSA_SRC)        = {masks}")
    if samples:
        print(f"samples (LSA_TISSUE_SRC) = {samples}")
    print(f"output (LSA_WORK)      = {work}")

    stages = resolve_stages(args.stages or None, has_samples=samples is not None)
    for num in stages:
        run_stage(num)
    print("\npipeline complete")
    print(f"interactive surfaces -> {os.path.join(work, 'model', 'lesion_3d_surfaces.html')}")


if __name__ == "__main__":
    main()
