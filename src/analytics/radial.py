"""
Radial mushrooming analyzer: coefficient of variation (CV) of lesion metrics
across radial increments from the tissue center.

Mushrooming biopsy stretches lesions toward the tissue fringes, so we expect
CV to increase with radius. This module analyzes CV (e.g. ablation diameter,
coagulation width) in radial bins and produces diagram and detailed analytics.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

# Re-export or use same scale as elsewhere
from src.constants import PIXEL_SCALE_UM

# Zone labels by radial position (bin index)
ZONE_LABELS = ['Central', 'Stable', 'Transition', 'Peripheral']


def _zone_for_bin(bin_index: int, n_bins: int) -> str:
    """Map bin index to zone: Central -> Stable -> Transition -> Peripheral."""
    if n_bins <= 0:
        return 'Central'
    q = (bin_index + 0.5) / n_bins  # 0–1
    if q <= 0.25:
        return 'Central'
    if q <= 0.5:
        return 'Stable'
    if q <= 0.75:
        return 'Transition'
    return 'Peripheral'


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation (std/mean). Returns 0 if mean is 0 or len < 2."""
    if len(values) < 2 or np.mean(values) == 0:
        return 0.0
    return float(np.std(values) / np.mean(values))


class RadialMushroomingAnalyzer:
    """
    Analyzes how coefficient of variation (CV) of lesion metrics changes
    with radial distance from the tissue center (mushrooming effect at edges).
    """

    def __init__(
        self,
        pixel_scale_um: float = PIXEL_SCALE_UM,
        n_radial_bins: int = 10,
    ):
        self.pixel_scale_um = pixel_scale_um
        self.n_radial_bins = n_radial_bins

    def _tissue_center(
        self,
        centers: List[Tuple[float, float]],
        mask_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        """Tissue center as centroid of lesion centers (or image center if no centers)."""
        if centers:
            arr = np.array(centers)
            return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
        if mask_shape:
            return (mask_shape[1] / 2.0, mask_shape[0] / 2.0)
        return (0.0, 0.0)

    def _radial_bins(
        self,
        centers: List[Tuple[float, float]],
        center_xy: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial distance (in µm) for each center and bin edges.

        Returns:
            radii_um: (n_lesions,) distances in µm.
            bin_edges_um: (n_bins+1,) edges in µm.
        """
        if not centers:
            return np.array([]), np.linspace(0, 1, self.n_radial_bins + 1)

        cx, cy = center_xy
        radii_px = np.array([
            np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in centers
        ])
        radii_um = radii_px * self.pixel_scale_um
        max_r = max(radii_um.max(), 1e-6)
        bin_edges_um = np.linspace(0, max_r, self.n_radial_bins + 1)
        return radii_um, bin_edges_um

    def analyze(
        self,
        centers: List[Tuple[float, float]],
        metrics_list: List[Any],
        mask_shape: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute CV of lesion metrics in each radial bin.

        Args:
            centers: List of (x, y) lesion centers (same order as metrics_list).
            metrics_list: List of objects with attributes:
                ablation_diameter_um, coagulation_width_um, lesion_diameter_um,
                ablation_area_um2, coagulation_area_um2 (at least diameter/width).
            mask_shape: Optional (H, W) for fallback center.

        Returns:
            DataFrame with columns:
                bin_index, bin_center_um, bin_min_um, bin_max_um, n_lesions,
                cv_ablation_diameter, cv_coagulation_width, cv_lesion_diameter,
                cv_ablation_area, cv_coagulation_area,
                mean_ablation_diameter_um, mean_coagulation_width_um, ...
        """
        center_xy = self._tissue_center(centers, mask_shape)
        radii_um, bin_edges_um = self._radial_bins(centers, center_xy)

        if len(centers) == 0 or len(metrics_list) == 0:
            return pd.DataFrame()

        # Extract metric arrays (support both attribute and dict-like)
        def _get(m, name, default=0.0):
            if hasattr(m, name):
                return getattr(m, name)
            if isinstance(m, dict):
                return m.get(name, default)
            return default

        ablation_d = np.array([_get(m, 'ablation_diameter_um') for m in metrics_list])
        coag_w = np.array([_get(m, 'coagulation_width_um') for m in metrics_list])
        lesion_d = np.array([_get(m, 'lesion_diameter_um') for m in metrics_list])
        ablation_a = np.array([_get(m, 'ablation_area_um2') for m in metrics_list])
        coag_a = np.array([_get(m, 'coagulation_area_um2') for m in metrics_list])

        rows = []
        for b in range(self.n_radial_bins):
            lo, hi = bin_edges_um[b], bin_edges_um[b + 1]
            in_bin = (radii_um >= lo) & (radii_um < hi) if b < self.n_radial_bins - 1 else (radii_um >= lo) & (radii_um <= hi)
            idx = np.where(in_bin)[0]
            n = len(idx)
            bin_center = (lo + hi) / 2.0

            row = {
                'bin_index': b,
                'bin_center_um': bin_center,
                'bin_min_um': lo,
                'bin_max_um': hi,
                'bin_lo_frac': b / self.n_radial_bins,
                'bin_hi_frac': (b + 1) / self.n_radial_bins,
                'bin_center_frac': (b + 0.5) / self.n_radial_bins,
                'n_lesions': n,
            }
            if n >= 2:
                row['cv_ablation_diameter'] = _cv(ablation_d[idx])
                row['cv_coagulation_width'] = _cv(coag_w[idx])
                row['cv_lesion_diameter'] = _cv(lesion_d[idx])
                row['cv_ablation_area'] = _cv(ablation_a[idx])
                row['cv_coagulation_area'] = _cv(coag_a[idx])
                row['mean_ablation_diameter_um'] = float(np.mean(ablation_d[idx]))
                row['mean_coagulation_width_um'] = float(np.mean(coag_w[idx]))
                row['mean_lesion_diameter_um'] = float(np.mean(lesion_d[idx]))
                row['std_ablation_diameter_um'] = float(np.std(ablation_d[idx]))
                row['std_coagulation_width_um'] = float(np.std(coag_w[idx]))
            else:
                for k in ['cv_ablation_diameter', 'cv_coagulation_width', 'cv_lesion_diameter',
                         'cv_ablation_area', 'cv_coagulation_area',
                         'mean_ablation_diameter_um', 'mean_coagulation_width_um', 'mean_lesion_diameter_um',
                         'std_ablation_diameter_um', 'std_coagulation_width_um']:
                    row[k] = np.nan if k.startswith('mean') or k.startswith('std') else 0.0
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_cv_vs_radius(
        self,
        df: pd.DataFrame,
        output_path: str,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Plot CV vs radial distance (bin center) for selected metrics.

        Args:
            df: Result of analyze().
            output_path: Path to save figure.
            metrics: Which CV columns to plot (default: ablation, coagulation width, lesion diameter).
        """
        if df.empty:
            return
        metrics = metrics or ['cv_ablation_diameter', 'cv_coagulation_width', 'cv_lesion_diameter']
        available = [m for m in metrics if m in df.columns]
        if not available:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        if 'bin_center_frac' in df.columns:
            x = df['bin_center_frac'] * 100.0
            xlabel = 'Normalized radius (% of max detected lesion radius)'
        else:
            x = df['bin_center_um']
            xlabel = 'Radial distance from lesion-centroid origin (µm)'
        for m in available:
            ax.plot(x, df[m], 'o-', label=m.replace('cv_', '').replace('_', ' ').title())
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Coefficient of variation')
        ax.set_title('CV vs radial bin (equal fractions of max detected radius)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def _get(self, m: Any, name: str, default: float = 0.0) -> float:
        if hasattr(m, name):
            return getattr(m, name)
        if isinstance(m, dict):
            return m.get(name, default)
        return default

    def plot_radial_mushrooming_diagram(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        centers: List[Tuple[float, float]],
        metrics_list: List[Any],
        df: pd.DataFrame,
        output_path: str,
    ) -> None:
        """
        Tissue overlay with concentric radial bins, lesion centers,
        and a single CV percentage label per bin.
        """
        if df.empty or not centers or not metrics_list:
            return
        cx, cy = self._tissue_center(centers, mask.shape[:2])
        bin_edges_um = np.concatenate(([df['bin_min_um'].iloc[0]], df['bin_max_um'].values))
        bin_edges_px = bin_edges_um / self.pixel_scale_um

        h, w = mask.shape[0], mask.shape[1]
        vis = np.ones((h, w, 4), dtype=np.float32)
        tissue = (mask == 1)
        ablation = (mask == 3)
        coag = (mask == 2)
        vis[tissue, :] = [0.2, 0.8, 0.2, 0.85]
        vis[ablation, :] = [0.9, 0.15, 0.15, 0.9]
        vis[coag, :] = [0.2, 0.3, 0.9, 0.9]
        vis[~(tissue | ablation | coag), :] = [1, 1, 1, 0.6]

        if image.shape[:2] == (h, w) and image.shape[2] >= 3:
            rgb = image[:, :, :3].astype(np.float32) / 255.0
            vis[:, :, :3] = 0.4 * vis[:, :, :3] + 0.6 * rgb
            vis[:, :, 3] = 0.92

        dpi = 200
        s = min(w, h) / 2000.0
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(vis)

        theta = np.linspace(0, 2 * np.pi, 200)
        for r_px in bin_edges_px[1:]:
            x_c = cx + r_px * np.cos(theta)
            y_c = cy + r_px * np.sin(theta)
            ax.plot(x_c, y_c, '--', color='gray', linewidth=max(1, 1.5 * s), alpha=0.8)

        ax.plot(cx, cy, 'r+', markersize=max(10, 18 * s), markeredgewidth=max(2, 4 * s))

        for x, y in centers:
            ax.plot(x, y, 'o', color='red', markersize=max(3, 5 * s))

        cv_col = 'cv_lesion_diameter' if 'cv_lesion_diameter' in df.columns else 'cv_coagulation_width'
        for b in range(len(df)):
            row = df.loc[df['bin_index'] == b, cv_col]
            if row.empty:
                continue
            cv_val = row.iloc[0]
            if pd.isna(cv_val) or cv_val <= 0:
                continue
            cv_pct = int(100 * cv_val)
            r_mid_px = (bin_edges_px[b] + bin_edges_px[b + 1]) / 2
            angle = np.pi / 4
            tx = cx + r_mid_px * np.cos(angle)
            ty = cy - r_mid_px * np.sin(angle)
            ax.text(tx, ty, f'{cv_pct}%', fontsize=max(5, 7 * s), color='darkgreen', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, pad=0.1 * s, lw=0.3))

        ax.axis('off')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=dpi, pad_inches=0)
        plt.close(fig)

    def plot_detailed_analytics(
        self,
        df: pd.DataFrame,
        output_path: str,
    ) -> None:
        """
        Multi-panel detailed analytics: CV vs radius with zone shading,
        mean metrics by bin, lesion count by bin, and zone summary.
        """
        if df.empty:
            return
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        if 'bin_center_frac' in df.columns:
            x = df['bin_center_frac'] * 100.0
            x_lo = df['bin_lo_frac'] * 100.0
            x_hi = df['bin_hi_frac'] * 100.0
            xlabel = 'Normalized radius (% of max detected lesion radius)'
            xmax = 102.0
        else:
            x = df['bin_center_um']
            x_lo = df['bin_min_um']
            x_hi = df['bin_max_um']
            xlabel = 'Radial distance (µm)'
            xmax = float(df['bin_center_um'].max()) * 1.02
        # Zone colors for shading
        zone_colors = {'Central': '#C8E6C9', 'Stable': '#A5D6A7', 'Transition': '#81C784', 'Peripheral': '#FFAB91'}

        # 1) CV vs radius with zone bands
        ax1 = fig.add_subplot(gs[0, 0])
        for bin_idx in range(len(df)):
            zone = _zone_for_bin(bin_idx, self.n_radial_bins)
            lo = x_lo.iloc[bin_idx]
            hi = x_hi.iloc[bin_idx]
            ax1.axvspan(lo, hi, alpha=0.35, color=zone_colors.get(zone, 'gray'))
        for col in ['cv_ablation_diameter', 'cv_coagulation_width', 'cv_lesion_diameter']:
            if col in df.columns:
                ax1.plot(x, df[col], 'o-', label=col.replace('cv_', '').replace('_', ' ').title(), linewidth=2)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Coefficient of variation')
        ax1.set_title('CV vs radial bin (zones shaded)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, xmax)

        # 2) Mean ablation diameter and coagulation width by bin
        ax2 = fig.add_subplot(gs[0, 1])
        width = float(np.mean(np.asarray(x_hi) - np.asarray(x_lo))) * 0.35
        if 'mean_ablation_diameter_um' in df.columns:
            ax2.bar(x - width * 0.55, df['mean_ablation_diameter_um'], width=width, label='Ablation diameter (µm)', color='coral', alpha=0.8)
        if 'mean_coagulation_width_um' in df.columns:
            ax2.bar(x + width * 0.55, df['mean_coagulation_width_um'], width=width, label='Coagulation width (µm)', color='steelblue', alpha=0.8)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Mean (µm)')
        ax2.set_title('Mean lesion metrics by radial bin')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3) Lesion count by bin
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(x, df['n_lesions'], width=(np.asarray(x_hi) - np.asarray(x_lo)) * 0.85,
                color='teal', alpha=0.7, edgecolor='black')
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel('Number of lesions')
        ax3.set_title('Lesion count by radial bin')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4) Zone summary: mean CV per zone
        ax4 = fig.add_subplot(gs[1, 1])
        zone_means = {}
        cv_col = 'cv_lesion_diameter' if 'cv_lesion_diameter' in df.columns else 'cv_coagulation_width'
        if cv_col not in df.columns:
            cv_col = [c for c in df.columns if c.startswith('cv_')][0] if any(c.startswith('cv_') for c in df.columns) else None
        if cv_col:
            for zone in ZONE_LABELS:
                bins_in_zone = [b for b in range(len(df)) if _zone_for_bin(b, self.n_radial_bins) == zone]
                vals = df.loc[df['bin_index'].isin(bins_in_zone), cv_col].dropna()
                vals = vals[vals > 0]
                zone_means[zone] = 100 * vals.mean() if len(vals) else 0
            zones = list(zone_means.keys())
            means = [zone_means[z] for z in zones]
            colors = [zone_colors.get(z, 'gray') for z in zones]
            ax4.barh(zones, means, color=colors, edgecolor='black')
            ax4.set_xlabel('Mean CV (%)')
            ax4.set_title('Mean CV by zone (Central → Peripheral)')
            ax4.grid(True, alpha=0.3, axis='x')
        plt.suptitle('Radial CV by bin — detailed analytics', fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
