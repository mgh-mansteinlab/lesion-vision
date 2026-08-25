"""
Peripheral enlargement vs radial position.

Used to answer reviewer requests:
  1. mean ± SD lesion size versus radius (not only CV)
  2. 28% peripheral-vs-central contrast with and without incomplete edge lesions
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Tuple, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from src.analytics.edge import (
    component_near_point,
    distance_to_background_px,
    labeled_lesion_components,
    lesion_edge_flags,
)
from src.constants import CLASS_ID_BACKGROUND, PERIPHERAL_EDGE_UM, PIXEL_SCALE_UM
from src.eval.metrics import rgb_to_class_mask

RadialOrigin = Literal["lesion_centroid", "punch_centroid"]
PunchXY = Mapping[str, Tuple[float, float]]

Image.MAX_IMAGE_PIXELS = None

_SAMPLE_RE = re.compile(r"^(?P<punch>.+)_sample_(?P<section>\d+)$")
_PUNCH_RE = re.compile(r"^(?P<punch>.+?)_(?P<section>\d+)$")
_SHORT_PUNCH_RE = re.compile(r"^([A-Za-z]+\d+)")


def punch_id_from_section(section_id: str) -> str:
    sample = _SAMPLE_RE.match(section_id)
    if sample:
        punch = sample.group("punch").strip()
        short = _SHORT_PUNCH_RE.match(punch)
        return short.group(1) if short else punch
    match = _PUNCH_RE.match(section_id)
    if match:
        return match.group("punch")
    return section_id


def load_lesion_summaries(predictions_dir: Path) -> pd.DataFrame:
    """Stack per-section lesion_summary.csv files under a prediction report."""
    rows: List[pd.DataFrame] = []
    for csv_path in sorted(predictions_dir.glob("*/csv/lesion_summary.csv")):
        section_id = csv_path.parents[1].name
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["section_id"] = section_id
        df["punch_id"] = punch_id_from_section(section_id)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No lesion_summary.csv under {predictions_dir}")
    out = pd.concat(rows, ignore_index=True)
    rename = {
        "Ablation_Diameter_um": "ablation_diameter_um",
        "Coagulation_Width_um": "coagulation_width_um",
        "Lesion_Diameter_um": "lesion_diameter_um",
        "Center_X_px": "center_x",
        "Center_Y_px": "center_y",
        "Touches_Edge": "touches_edge",
        "Dist_To_Edge_um": "dist_to_edge_um",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "center_x" not in out.columns and "Center_X" in out.columns:
        out["center_x"] = out["Center_X"]
    if "center_y" not in out.columns and "Center_Y" in out.columns:
        out["center_y"] = out["Center_Y"]
    return out


def annotate_edge_from_pred_png(
    df: pd.DataFrame,
    predictions_dir: Path,
    pixel_scale_um: float = PIXEL_SCALE_UM,
    search_px: int = 40,
) -> pd.DataFrame:
    """Add touches_edge and dist_to_edge_um using each section's *_pred.png."""
    annotated: List[pd.DataFrame] = []
    for section_id, part in df.groupby("section_id", sort=True):
        pred_path = predictions_dir / section_id / "images" / f"{section_id}_pred.png"
        chunk = part.copy()
        if not pred_path.is_file():
            chunk["touches_edge"] = False
            chunk["dist_to_edge_um"] = np.nan
            chunk["edge_matched"] = False
            annotated.append(chunk)
            continue
        rgb = np.asarray(Image.open(pred_path).convert("RGB"))
        mask = rgb_to_class_mask(rgb)
        dist_px = distance_to_background_px(mask)
        _binary, labels = labeled_lesion_components(mask)
        touches_col: List[bool] = []
        dist_col: List[float] = []
        matched_col: List[bool] = []
        for _, row in chunk.iterrows():
            component = component_near_point(
                labels, float(row["center_x"]), float(row["center_y"]), search_px=search_px
            )
            if component is None:
                touches_col.append(False)
                dist_col.append(float("nan"))
                matched_col.append(False)
                continue
            touches, dist_um = lesion_edge_flags(
                component.astype(np.uint8),
                mask,
                dist_px=dist_px,
                pixel_scale_um=pixel_scale_um,
            )
            touches_col.append(touches)
            dist_col.append(dist_um)
            matched_col.append(True)
        chunk["touches_edge"] = touches_col
        chunk["dist_to_edge_um"] = dist_col
        chunk["edge_matched"] = matched_col
        annotated.append(chunk)
    return pd.concat(annotated, ignore_index=True)


def punch_centroid_from_pred_png(
    pred_path: Path,
    downsample: int = 8,
) -> Optional[Tuple[float, float]]:
    """Tissue-pixel centroid in full pred-image coordinates (x, y)."""
    if not pred_path.is_file():
        return None
    image = Image.open(pred_path).convert("RGB")
    width, height = image.size
    step = max(1, int(downsample))
    small = image.resize(
        (max(1, width // step), max(1, height // step)),
        Image.Resampling.NEAREST,
    )
    mask = rgb_to_class_mask(np.asarray(small))
    tissue = mask != CLASS_ID_BACKGROUND
    if not np.any(tissue):
        return None
    ys, xs = np.where(tissue)
    cx = (float(xs.mean()) + 0.5) * step
    cy = (float(ys.mean()) + 0.5) * step
    return cx, cy


def punch_centroids_from_pred_pngs(
    section_ids: List[str],
    predictions_dir: Path,
    downsample: int = 8,
) -> Dict[str, Tuple[float, float]]:
    """Map section_id → (cx, cy) from each section's ``*_pred.png``."""
    out: Dict[str, Tuple[float, float]] = {}
    for section_id in section_ids:
        pred_path = predictions_dir / section_id / "images" / f"{section_id}_pred.png"
        xy = punch_centroid_from_pred_png(pred_path, downsample=downsample)
        if xy is not None:
            out[section_id] = xy
    return out


def add_radial_position(
    df: pd.DataFrame,
    pixel_scale_um: float = PIXEL_SCALE_UM,
    origin: RadialOrigin = "lesion_centroid",
    punch_xy: Optional[PunchXY] = None,
) -> pd.DataFrame:
    """Distance from a per-section origin (µm and % of max detected radius)."""
    out = df.copy()
    out["radius_um"] = np.nan
    out["radius_frac"] = np.nan
    out["origin_x"] = np.nan
    out["origin_y"] = np.nan
    for section_id, part in out.groupby("section_id"):
        sid = str(section_id)
        if origin == "lesion_centroid":
            cx = float(part["center_x"].mean())
            cy = float(part["center_y"].mean())
        elif origin == "punch_centroid":
            if punch_xy is None or sid not in punch_xy:
                continue
            cx, cy = punch_xy[sid]
        else:
            unreachable = cast(RadialOrigin, origin)
            raise ValueError(f"unknown radial origin: {unreachable}")
        dx = part["center_x"].to_numpy(dtype=float) - cx
        dy = part["center_y"].to_numpy(dtype=float) - cy
        r_um = np.hypot(dx, dy) * pixel_scale_um
        max_r = float(np.max(r_um)) if len(r_um) else 0.0
        out.loc[part.index, "origin_x"] = cx
        out.loc[part.index, "origin_y"] = cy
        out.loc[part.index, "radius_um"] = r_um
        out.loc[part.index, "radius_frac"] = r_um / max_r if max_r > 0 else 0.0
    return out


def _filter_complete(df: pd.DataFrame, drop_edge: bool) -> pd.DataFrame:
    if not drop_edge:
        return df
    if "touches_edge" not in df.columns:
        return df
    return df.loc[~df["touches_edge"].fillna(False).astype(bool)].copy()


def paired_peripheral_contrast(
    df: pd.DataFrame,
    value_col: str = "ablation_diameter_um",
    group_col: str = "section_id",
    edge_um: float = PERIPHERAL_EDGE_UM,
    drop_edge: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Within-group mean(peripheral) vs mean(central).

    Peripheral = min distance to background ≤ edge_um (manuscript 500 µm).
    Central = farther than edge_um from background.
    """
    work = _filter_complete(df, drop_edge)
    work = work.dropna(subset=[value_col, "dist_to_edge_um"])
    rows = []
    for gid, part in work.groupby(group_col):
        peri = part.loc[part["dist_to_edge_um"] <= edge_um, value_col]
        cent = part.loc[part["dist_to_edge_um"] > edge_um, value_col]
        if len(peri) < 1 or len(cent) < 1:
            continue
        cmean = float(cent.mean())
        pmean = float(peri.mean())
        if cmean == 0:
            continue
        rows.append({
            group_col: gid,
            "n_central": int(len(cent)),
            "n_peripheral": int(len(peri)),
            "mean_central": cmean,
            "mean_peripheral": pmean,
            "pct_increase": 100.0 * (pmean - cmean) / cmean,
        })
    paired = pd.DataFrame(rows)
    summary: Dict[str, float] = {
        "n_groups": float(len(paired)),
        "mean_pct_increase": float("nan"),
        "sd_pct_increase": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "n_lesions": float(len(work)),
        "n_edge_dropped": float(int(df["touches_edge"].fillna(False).sum()) if drop_edge and "touches_edge" in df.columns else 0),
    }
    if len(paired) >= 2:
        summary["mean_pct_increase"] = float(paired["pct_increase"].mean())
        summary["sd_pct_increase"] = float(paired["pct_increase"].std(ddof=1))
        t_stat, p_val = stats.ttest_rel(paired["mean_peripheral"], paired["mean_central"])
        summary["t_stat"] = float(t_stat)
        summary["p_value"] = float(p_val)
    return paired, summary


def mean_size_vs_radius(
    df: pd.DataFrame,
    value_col: str = "ablation_diameter_um",
    n_bins: int = 10,
    drop_edge: bool = False,
    group_col: str = "section_id",
) -> pd.DataFrame:
    """
    For each normalized-radius bin, take the per-section mean, then mean ± SD
    across sections (so the error bar is not inflated by lesion count).
    """
    work = _filter_complete(df, drop_edge)
    work = work.dropna(subset=[value_col, "radius_frac"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            in_bin = (work["radius_frac"] >= lo) & (work["radius_frac"] < hi)
        else:
            in_bin = (work["radius_frac"] >= lo) & (work["radius_frac"] <= hi)
        sub = work.loc[in_bin]
        per_group = sub.groupby(group_col)[value_col].mean()
        n_groups = int(per_group.shape[0])
        rows.append({
            "bin_index": b,
            "bin_lo_frac": lo,
            "bin_hi_frac": hi,
            "bin_center_frac": 0.5 * (lo + hi),
            "n_lesions": int(len(sub)),
            "n_groups": n_groups,
            "mean_um": float(per_group.mean()) if n_groups else float("nan"),
            "sd_um": float(per_group.std(ddof=1)) if n_groups >= 2 else float("nan"),
            "mean_radius_um": float(sub["radius_um"].mean()) if len(sub) else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_mean_vs_radius(
    bin_df: pd.DataFrame,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = bin_df["bin_center_frac"] * 100.0
    y = bin_df["mean_um"]
    err = bin_df["sd_um"].fillna(0.0)
    ax.errorbar(x, y, yerr=err, fmt="o-", capsize=4, linewidth=1.5, markersize=6)
    ax.set_xlabel("Radial position (% of max detected radius in the section)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paired_contrast(
    paired: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if paired.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        np.zeros(len(paired)),
        paired["mean_central"],
        "o",
        color="#4C78A8",
        alpha=0.8,
        label="Central (>{:.0f} µm from edge)".format(PERIPHERAL_EDGE_UM),
    )
    ax.plot(
        np.ones(len(paired)),
        paired["mean_peripheral"],
        "o",
        color="#E45756",
        alpha=0.8,
        label="Peripheral (≤{:.0f} µm from edge)".format(PERIPHERAL_EDGE_UM),
    )
    for _, row in paired.iterrows():
        ax.plot(
            [0, 1],
            [row["mean_central"], row["mean_peripheral"]],
            color="#9E9E9E",
            alpha=0.5,
            linewidth=0.8,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Central", "Peripheral"])
    ax.set_ylabel("Mean ablation diameter (µm)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_contrast_line(label: str, summary: Dict[str, float]) -> str:
    n = int(summary["n_groups"])
    if n < 2:
        return f"{label}: not enough paired groups (n={n})"
    return (
        f"{label}: {summary['mean_pct_increase']:.1f} ± {summary['sd_pct_increase']:.1f}% "
        f"(n={n} groups, paired t={summary['t_stat']:.2f}, p={summary['p_value']:.4g})"
    )
