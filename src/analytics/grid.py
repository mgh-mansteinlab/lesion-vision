"""
Grid mapper: infers grid structure from lesion centers and computes spacing.
Used for biopsy grid alignment and distance metrics.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from src.constants import PIXEL_SCALE_UM


class GridMapper:
    """
    Infers grid lines from lesion centers and computes adjacent-lesion distances
    and grid visualizations.
    """

    def __init__(self, pixel_scale_um: float = PIXEL_SCALE_UM):
        self.pixel_scale_um = pixel_scale_um

    def get_grid_lines(
        self,
        centers: List[Tuple[float, float]],
        gap_multiplier: float = 1.5,
    ) -> Tuple[List[float], List[float]]:
        """
        Infer x and y grid lines from lesion center coordinates using gaps.

        Args:
            centers: List of (x, y) lesion centers in pixels.
            gap_multiplier: Multiplier on median gap to define a new grid line.

        Returns:
            (x_grid_lines, y_grid_lines) in pixel coordinates.
        """
        if len(centers) < 2:
            return [], []

        centers = np.array(centers)
        x_sorted = np.sort(centers[:, 0])
        y_sorted = np.sort(centers[:, 1])
        x_gaps = np.diff(x_sorted)
        y_gaps = np.diff(y_sorted)
        x_threshold = np.median(x_gaps) * gap_multiplier
        y_threshold = np.median(y_gaps) * gap_multiplier

        x_grid_lines = [float(x_sorted[0])]
        current_x = x_sorted[0]
        for i in range(1, len(x_sorted)):
            if x_sorted[i] - current_x > x_threshold:
                x_grid_lines.append(float(x_sorted[i]))
                current_x = x_sorted[i]

        y_grid_lines = [float(y_sorted[0])]
        current_y = y_sorted[0]
        for i in range(1, len(y_sorted)):
            if y_sorted[i] - current_y > y_threshold:
                y_grid_lines.append(float(y_sorted[i]))
                current_y = y_sorted[i]

        return x_grid_lines, y_grid_lines

    def calculate_adjacent_distances_improved(
        self, centers: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Mean nearest-neighbor distance and median as grid spacing estimate.

        Args:
            centers: List of (x, y) lesion centers in pixels.

        Returns:
            Dict with 'mean_nearest_neighbor_um' and 'grid_spacing_um'.
        """
        if len(centers) < 2:
            return {'mean_nearest_neighbor_um': 0.0, 'grid_spacing_um': 0.0}

        centers_array = np.array(centers)
        n = len(centers_array)
        nearest = []
        for i in range(n):
            dists = np.sqrt(
                (centers_array[i, 0] - centers_array[:, 0]) ** 2
                + (centers_array[i, 1] - centers_array[:, 1]) ** 2
            )
            dists[i] = np.inf
            nearest.append(np.min(dists))
        nearest = np.array(nearest)
        scale = self.pixel_scale_um
        return {
            'mean_nearest_neighbor_um': float(np.mean(nearest) * scale),
            'grid_spacing_um': float(np.median(nearest) * scale),
        }

    def calculate_adjacent_distances(
        self, centers: List[Tuple[float, float]]
    ) -> float:
        """
        Mean distance between adjacent lesions along grid (horizontal/vertical).
        Returns value in micrometers.
        """
        if len(centers) < 2:
            return 0.0

        centers = np.array(centers)
        x_grid_lines, y_grid_lines = self.get_grid_lines(centers.tolist())
        if not x_grid_lines or not y_grid_lines:
            return 0.0

        distances = []
        for i in range(len(centers)):
            x_pos = min(
                range(len(x_grid_lines)),
                key=lambda j: abs(centers[i, 0] - x_grid_lines[j]),
            )
            y_pos = min(
                range(len(y_grid_lines)),
                key=lambda j: abs(centers[i, 1] - y_grid_lines[j]),
            )
            for j in range(len(centers)):
                if i == j:
                    continue
                nx_pos = min(
                    range(len(x_grid_lines)),
                    key=lambda k: abs(centers[j, 0] - x_grid_lines[k]),
                )
                ny_pos = min(
                    range(len(y_grid_lines)),
                    key=lambda k: abs(centers[j, 1] - y_grid_lines[k]),
                )
                if ny_pos == y_pos and abs(nx_pos - x_pos) == 1:
                    distances.append(abs(centers[i, 0] - centers[j, 0]))
                if nx_pos == x_pos and abs(ny_pos - y_pos) == 1:
                    distances.append(abs(centers[i, 1] - centers[j, 1]))

        if not distances:
            return 0.0
        return float(np.mean(distances) * self.pixel_scale_um)

    def create_grid_visualization(
        self,
        image: np.ndarray,
        centers: List[Tuple[float, float]],
        output_path: str,
    ) -> None:
        """
        Draw inferred grid lines on the image and save.

        Args:
            image: RGB image (H, W, 3).
            centers: List of (x, y) lesion centers in pixels.
            output_path: Path to save the figure.
        """
        h, w = image.shape[:2]
        dpi = 200
        scale = min(w, h) / 2000.0

        x_grid_lines, y_grid_lines = self.get_grid_lines(centers)
        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(image)
        for x in x_grid_lines:
            ax.axvline(x=x, color='yellow', linestyle='--', alpha=0.5)
        for y in y_grid_lines:
            ax.axhline(y=y, color='yellow', linestyle='--', alpha=0.5)
        for cx, cy in centers:
            ax.plot(cx, cy, 'r+', markersize=max(8, 10 * scale))
        ax.text(0.5, 0.99, 'Grid Structure', transform=ax.transAxes,
                ha='center', va='top', fontsize=max(10, 14 * scale), fontweight='bold')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, dpi=dpi, pad_inches=0)
        plt.close(fig)
