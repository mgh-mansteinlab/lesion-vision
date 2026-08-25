"""Tests for incomplete-boundary flags and peripheral-vs-central contrast."""

import numpy as np
import pandas as pd

from src.analytics.edge import lesion_binary, lesion_edge_flags
from src.analytics.lesions import LesionAnalyzer
from src.analytics.peripheral import (
    add_radial_position,
    mean_size_vs_radius,
    paired_peripheral_contrast,
    punch_id_from_section,
)
from src.constants import CLASS_ID_ABLATION, CLASS_ID_COAGULATION, CLASS_ID_TISSUE


def _disk(h, w, cy, cx, radius):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def test_punch_id_from_section():
    assert punch_id_from_section("PVcont1_01") == "PVcont1"
    assert punch_id_from_section("PVcont18_03") == "PVcont18"
    assert punch_id_from_section("CPBSV10 - 2020-10-08 14.57.10_sample_01") == "CPBSV10"
    assert punch_id_from_section("CPVV1 - 2020-10-08 15.14.54_sample_03") == "CPVV1"


def test_edge_lesion_touches_background_central_does_not():
    h = w = 80
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[_disk(h, w, 40, 40, 28)] = CLASS_ID_TISSUE

    central = np.zeros((h, w), dtype=np.uint8)
    central[_disk(h, w, 40, 40, 8)] = 1
    mask[central > 0] = CLASS_ID_COAGULATION
    mask[_disk(h, w, 40, 40, 3)] = CLASS_ID_ABLATION

    edge = np.zeros((h, w), dtype=np.uint8)
    edge[_disk(h, w, 40, 68, 8)] = 1
    mask[edge > 0] = CLASS_ID_COAGULATION

    c_touch, c_dist = lesion_edge_flags(central, mask)
    e_touch, e_dist = lesion_edge_flags(edge, mask)
    assert c_touch is False
    assert e_touch is True
    assert e_dist < c_dist
    assert np.any(lesion_binary(mask))


def test_analyze_mask_writes_edge_columns(tmp_path):
    h = w = 64
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[_disk(h, w, 32, 32, 24)] = CLASS_ID_TISSUE
    mask[_disk(h, w, 32, 32, 12)] = CLASS_ID_COAGULATION
    mask[_disk(h, w, 32, 32, 5)] = CLASS_ID_ABLATION
    image = np.zeros((h, w, 3), dtype=np.uint8)
    analyzer = LesionAnalyzer()
    df = analyzer.analyze_mask(mask, image, str(tmp_path), summary_metrics=True, figures=False)
    assert "Touches_Edge" in df.columns
    assert "Dist_To_Edge_um" in df.columns
    assert len(df) >= 1
    summary = pd.read_csv(tmp_path / "lesion_summary.csv")
    assert "Touches_Edge" in summary.columns


def test_paired_contrast_and_radius_bins():
    rows = []
    for section in ("A_01", "B_01", "C_01"):
        for i in range(6):
            rows.append({
                "section_id": section,
                "punch_id": section.split("_")[0],
                "ablation_diameter_um": 80.0 if i < 4 else 104.0,
                "center_x": 10.0 + i,
                "center_y": 10.0,
                "dist_to_edge_um": 800.0 if i < 4 else 100.0,
                "touches_edge": i == 5,
            })
    df = pd.DataFrame(rows)
    df = add_radial_position(df, pixel_scale_um=1.0)
    paired, summary = paired_peripheral_contrast(df, drop_edge=False)
    assert summary["n_groups"] == 3
    assert summary["mean_pct_increase"] == 30.0
    assert summary["p_value"] < 0.05

    paired_drop, summary_drop = paired_peripheral_contrast(df, drop_edge=True)
    assert summary_drop["n_edge_dropped"] == 3
    assert summary_drop["n_groups"] == 3
    assert abs(summary_drop["mean_pct_increase"] - 30.0) < 1e-6

    bins = mean_size_vs_radius(df, n_bins=4, drop_edge=False)
    assert len(bins) == 4
    assert bins["n_lesions"].sum() == len(df)


def test_punch_centroid_origin_shifts_radius():
    df = pd.DataFrame({
        "section_id": ["S1", "S1", "S1"],
        "center_x": [0.0, 10.0, 20.0],
        "center_y": [0.0, 0.0, 0.0],
        "ablation_diameter_um": [50.0, 50.0, 50.0],
        "dist_to_edge_um": [800.0, 800.0, 800.0],
        "touches_edge": [False, False, False],
    })
    lesion = add_radial_position(df, pixel_scale_um=1.0, origin="lesion_centroid")
    punch = add_radial_position(
        df,
        pixel_scale_um=1.0,
        origin="punch_centroid",
        punch_xy={"S1": (0.0, 0.0)},
    )
    assert abs(float(lesion["origin_x"].iloc[0]) - 10.0) < 1e-9
    assert abs(float(punch["origin_x"].iloc[0]) - 0.0) < 1e-9
    assert float(punch["radius_um"].max()) > float(lesion["radius_um"].max())
    empty = add_radial_position(df, origin="punch_centroid", punch_xy={})
    assert empty["radius_um"].isna().all()
