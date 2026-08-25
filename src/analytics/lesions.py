"""
Lesion analytics: metrics, lesion detection, and analysis on segmentation masks.
Separate from prediction; use LesionAnalyzer on a mask (e.g. from LesionPredictor.predict()).
"""
import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analytics.grid import GridMapper
from src.analytics.edge import distance_to_background_px, lesion_edge_flags
from src.constants import PIXEL_SCALE_UM


@dataclass
class LesionMetrics:
    ablation_diameter_um: float
    coagulation_width_um: float
    ablation_area_um2: float
    coagulation_area_um2: float
    center: Tuple[float, float]
    lesion_diameter_um: float
    touches_edge: bool = False
    dist_to_edge_um: float = float("nan")

    @classmethod
    def from_pixels(
        cls,
        ablation_diameter_px: float,
        coagulation_width_px: float,
        ablation_area_px: float,
        coagulation_area_px: float,
        center: Tuple[float, float],
        lesion_diameter_px: float,
        pixel_scale_um: float = PIXEL_SCALE_UM,
    ) -> 'LesionMetrics':
        """Create LesionMetrics from pixel measurements."""
        scale = pixel_scale_um
        return cls(
            ablation_diameter_um=ablation_diameter_px * scale,
            coagulation_width_um=coagulation_width_px * scale,
            ablation_area_um2=ablation_area_px * (scale ** 2),
            coagulation_area_um2=coagulation_area_px * (scale ** 2),
            center=center,
            lesion_diameter_um=lesion_diameter_px * scale,
        )


class LesionAnalyzer:
    """
    Analyzes a segmentation mask: finds lesions, computes metrics, outlier filtering,
    and visualizations. Uses GridMapper for grid-related computations.
    """

    def __init__(self):
        self.metrics_list: List[LesionMetrics] = []
        self.lesion_centers: List[Tuple[float, float]] = []
        self._grid_mapper = GridMapper(pixel_scale_um=PIXEL_SCALE_UM)

    def find_lesions(self, mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
        """
        Find individual lesions directly from the full prediction mask.
        Mask: 1=Tissue, 2=Coagulation, 3=Ablation.
        Returns list of (ablation_crop, coagulation_crop, (x1, y1, x2, y2)) per lesion.
        Crops are local to the bounding box; bbox maps back to full-image coordinates.
        """
        lesion_binary = ((mask == 2) | (mask == 3)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_closed = cv2.morphologyEx(lesion_binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(lesion_closed)

        lesions = []
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            if cv2.countNonZero(component) < 100:
                continue

            ys, xs = np.where(component > 0)
            x1, x2 = int(np.min(xs)), int(np.max(xs)) + 1
            y1, y2 = int(np.min(ys)), int(np.max(ys)) + 1

            region_mask = mask[y1:y2, x1:x2]
            component_crop = component[y1:y2, x1:x2]

            ablation = ((region_mask == 3) & (component_crop > 0)).astype(np.uint8)
            coagulation = ((region_mask == 2) & (component_crop > 0)).astype(np.uint8)

            if cv2.countNonZero(coagulation) < 100:
                continue

            lesions.append((ablation, coagulation, (x1, y1, x2, y2)))

        return lesions

    def calculate_metrics(self, ablation: np.ndarray, coagulation: np.ndarray) -> LesionMetrics:
        """Compute LesionMetrics for one lesion (ablation + coagulation masks)."""
        has_ablation = np.any(ablation)
        if has_ablation:
            ablation_contours = cv2.findContours(ablation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if ablation_contours:
                ablation_contour = ablation_contours[0]
                ablation_center = np.mean(ablation_contour, axis=0)[0]
                ablation_diameter_px = 2 * np.sqrt(cv2.contourArea(ablation_contour) / np.pi)
            else:
                has_ablation = False
        if not has_ablation:
            coag_contours = cv2.findContours(coagulation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if coag_contours:
                coag_contour = coag_contours[0]
                ablation_center = np.mean(coag_contour, axis=0)[0]
            else:
                moments = cv2.moments(coagulation)
                if moments['m00'] != 0:
                    ablation_center = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
                else:
                    ablation_center = np.array([coagulation.shape[1]//2, coagulation.shape[0]//2])
            ablation_diameter_px = 0

        combined = cv2.bitwise_or(ablation, coagulation)
        outer_contours = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        lesion_diameter_px = 0
        if outer_contours:
            outer_contour = outer_contours[0]
            max_distance = 0
            for point in outer_contour:
                pt = point[0]
                distance = np.linalg.norm(pt - ablation_center)
                max_distance = max(max_distance, distance)
            lesion_diameter_px = 2 * max_distance

        num_samples = 72
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        widths = []
        for angle in angles:
            ray_length = max(combined.shape)
            ray_end = ablation_center + ray_length * np.array([np.cos(angle), np.sin(angle)])
            outer_intersection = None
            min_dist = float('inf')
            for point in outer_contour:
                pt = point[0]
                ray_dir = ray_end - ablation_center
                point_dir = pt - ablation_center
                if np.dot(ray_dir, point_dir) > 0:
                    dist = np.linalg.norm(pt - ablation_center)
                    if dist < min_dist:
                        min_dist = dist
                        outer_intersection = pt
            if outer_intersection is not None:
                if has_ablation:
                    ablation_boundary = cv2.findContours(ablation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                    ablation_intersection = None
                    min_dist = float('inf')
                    for point in ablation_boundary:
                        pt = point[0]
                        ray_dir = ray_end - ablation_center
                        point_dir = pt - ablation_center
                        if np.dot(ray_dir, point_dir) > 0:
                            dist = np.linalg.norm(pt - ablation_center)
                            if dist < min_dist:
                                min_dist = dist
                                ablation_intersection = pt
                    if ablation_intersection is not None:
                        widths.append(np.linalg.norm(outer_intersection - ablation_intersection))
                else:
                    widths.append(np.linalg.norm(outer_intersection - ablation_center))
        coagulation_width_px = np.median(widths) if widths else 0
        ablation_area_px = np.sum(ablation)
        coagulation_area_px = np.sum(coagulation)
        plt.close('all')
        return LesionMetrics.from_pixels(
            ablation_diameter_px=ablation_diameter_px,
            coagulation_width_px=coagulation_width_px,
            ablation_area_px=ablation_area_px,
            coagulation_area_px=coagulation_area_px,
            center=tuple(ablation_center),
            lesion_diameter_px=lesion_diameter_px,
        )

    def create_lesion_plot(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        lesion_id: int,
        ablation: np.ndarray,
        coagulation: np.ndarray,
        metrics: LesionMetrics,
        output_path: str,
    ) -> None:
        """Save a 2x2 detailed plot for one lesion."""
        combined = cv2.bitwise_or(ablation, coagulation)
        y_indices, x_indices = np.where(combined > 0)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        padding = 50
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(image[y_min:y_max, x_min:x_max])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        colored_mask = np.zeros((*mask[y_min:y_max, x_min:x_max].shape, 3), dtype=np.uint8)
        colored_mask[mask[y_min:y_max, x_min:x_max] == 1] = [0, 255, 0]
        colored_mask[mask[y_min:y_max, x_min:x_max] == 2] = [0, 0, 255]
        colored_mask[mask[y_min:y_max, x_min:x_max] == 3] = [255, 0, 0]
        axes[0, 1].imshow(colored_mask)
        axes[0, 1].set_title('Segmentation')
        axes[0, 1].axis('off')
        axes[1, 0].imshow(ablation[y_min:y_max, x_min:x_max], cmap='Reds')
        axes[1, 0].set_title('Ablation Zone')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(coagulation[y_min:y_max, x_min:x_max], cmap='Blues')
        axes[1, 1].set_title('Coagulation Zone')
        axes[1, 1].axis('off')
        fig.text(0.5, 0.02, (
            f"Lesion {lesion_id} Metrics:\n"
            f"Ablation Diameter: {metrics.ablation_diameter_um:.1f} μm\n"
            f"Coagulation Width: {metrics.coagulation_width_um:.1f} μm\n"
            f"Ablation Area: {metrics.ablation_area_um2:.1f} μm²\n"
            f"Coagulation Area: {metrics.coagulation_area_um2:.1f} μm²"
        ), ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def create_lesion_crop_figure(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        lesion_id: int,
        metrics: LesionMetrics,
        output_path: str,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        """Save a single cropped lesion figure with semi-transparent overlays.

        Crops directly from the full prediction mask — no per-tile masks needed.
        """
        from matplotlib.patches import Patch

        bx1, by1, bx2, by2 = bbox
        padding = 100
        crop_y1 = max(0, by1 - padding)
        crop_y2 = min(image.shape[0], by2 + padding)
        crop_x1 = max(0, bx1 - padding)
        crop_x2 = min(image.shape[1], bx2 + padding)

        crop_h = crop_y2 - crop_y1
        crop_w = crop_x2 - crop_x1

        cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_ablation = (cropped_mask == 3).astype(np.uint8)
        cropped_coagulation = (cropped_mask == 2).astype(np.uint8)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        ax.imshow(cropped_image)

        overlay = np.zeros((crop_h, crop_w, 4), dtype=np.float32)
        coag_mask = cropped_coagulation > 0
        overlay[coag_mask] = [0.2, 0.4, 1.0, 0.35]
        abl_mask = cropped_ablation > 0
        overlay[abl_mask] = [1.0, 0.2, 0.2, 0.4]
        ax.imshow(overlay)

        if np.any(cropped_ablation):
            contours = cv2.findContours(cropped_ablation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and contour.shape[0] > 2:
                    closed = np.vstack([contour, contour[0]])
                    ax.plot(closed[:, 0], closed[:, 1], color='#cc0000', linewidth=2)
        if np.any(cropped_coagulation):
            contours = cv2.findContours(cropped_coagulation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and contour.shape[0] > 2:
                    closed = np.vstack([contour, contour[0]])
                    ax.plot(closed[:, 0], closed[:, 1], color='#0044cc', linewidth=2)

        center_x = metrics.center[0] - crop_x1
        center_y = metrics.center[1] - crop_y1
        ax.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=3)

        legend_elements = [
            Patch(facecolor=(1.0, 0.2, 0.2, 0.4), edgecolor='#cc0000', linewidth=2, label='Ablation'),
            Patch(facecolor=(0.2, 0.4, 1.0, 0.35), edgecolor='#0044cc', linewidth=2, label='Coagulation'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        has_ablation = metrics.ablation_diameter_um > 0
        title = (f"Lesion {lesion_id}: Coag={metrics.coagulation_width_um:.1f}\u03bcm, Abl={metrics.ablation_diameter_um:.1f}\u03bcm\n"
                 f"CoagArea={metrics.coagulation_area_um2:.0f}\u03bcm\u00b2, AblArea={metrics.ablation_area_um2:.0f}\u03bcm\u00b2") if has_ablation else (
                 f"Lesion {lesion_id}: Coag={metrics.coagulation_width_um:.1f}\u03bcm (No Ablation)\nCoagArea={metrics.coagulation_area_um2:.0f}\u03bcm\u00b2")
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig)

    def calculate_adjacent_distances_improved(self, centers: List[Tuple[float, float]]) -> Dict[str, float]:
        """Delegate to GridMapper."""
        return self._grid_mapper.calculate_adjacent_distances_improved(centers)

    def calculate_adjacent_distances(self, centers: List[Tuple[float, float]]) -> float:
        """Delegate to GridMapper (mean adjacent grid distance in µm)."""
        return self._grid_mapper.calculate_adjacent_distances(centers)

    def filter_outliers(
        self,
        metrics_list: List[LesionMetrics],
        centers: List[Tuple[float, float]],
    ) -> List[int]:
        """Return indices of valid (non-outlier) lesions."""
        if not metrics_list:
            return []
        valid_indices = []
        coag_areas = [m.coagulation_area_um2 for m in metrics_list]
        coag_widths = [m.coagulation_width_um for m in metrics_list]
        coag_area_median = np.median(coag_areas)
        coag_width_median = np.median(coag_widths)
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2) * PIXEL_SCALE_UM
                distances.append(dist)
        avg_distance = np.median(distances) if distances else 1000
        min_area_threshold = 25
        min_width_threshold = 1.0
        for i, metrics in enumerate(metrics_list):
            is_valid = True
            if metrics.coagulation_area_um2 < min_area_threshold:
                is_valid = False
            if metrics.coagulation_width_um < min_width_threshold:
                is_valid = False
            if is_valid:
                valid_indices.append(i)
        return valid_indices

    def analyze_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        output_dir: str,
        summary_metrics: bool = False,
        figures: bool = False,
        figures_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pipeline: find lesions → metrics → filter → IDs/grid/CSVs → crops.
        Sets self.metrics_list, self.lesion_centers.
        Returns DataFrame of per-lesion metrics.

        Args:
            figures_dir: Directory for lesion crop figures. When None, defaults
                         to ``output_dir/figures``.
        """
        self.metrics_list = []
        self.lesion_centers = []
        if figures:
            if figures_dir is None:
                figures_dir = os.path.join(output_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)

        # 1. Find lesions from the full prediction mask
        lesions = self.find_lesions(mask)

        # 2. Compute per-lesion metrics (distance / area)
        dist_px = distance_to_background_px(mask)
        raw_metrics_list = []
        raw_centers = []
        raw_bboxes = []
        for ablation, coagulation, bbox in lesions:
            x1, y1, x2, y2 = bbox
            metrics = self.calculate_metrics(ablation, coagulation)
            metrics.center = (metrics.center[0] + x1, metrics.center[1] + y1)
            component = np.zeros(mask.shape[:2], dtype=np.uint8)
            component[y1:y2, x1:x2] = ((ablation > 0) | (coagulation > 0)).astype(np.uint8)
            touches, dist_um = lesion_edge_flags(component, mask, dist_px=dist_px)
            metrics.touches_edge = touches
            metrics.dist_to_edge_um = dist_um
            raw_metrics_list.append(metrics)
            raw_centers.append(metrics.center)
            raw_bboxes.append(bbox)

        # 3. ID assignment: filter outliers, assign sequential IDs
        filtered_indices = self.filter_outliers(raw_metrics_list, raw_centers)
        self.metrics_list = [raw_metrics_list[i] for i in filtered_indices]
        self.lesion_centers = [raw_centers[i] for i in filtered_indices]
        filtered_bboxes = [raw_bboxes[i] for i in filtered_indices]

        # Build per-lesion DataFrame and grid/distance metrics
        lesions_data = []
        for i, metrics in enumerate(self.metrics_list, 1):
            lesions_data.append({
                'Lesion_ID': i,
                'Ablation_Diameter_um': metrics.ablation_diameter_um,
                'Coagulation_Width_um': metrics.coagulation_width_um,
                'Lesion_Diameter_um': metrics.lesion_diameter_um,
                'Ablation_Area_um2': metrics.ablation_area_um2,
                'Coagulation_Area_um2': metrics.coagulation_area_um2,
                'Center_X': metrics.center[0],
                'Center_Y': metrics.center[1],
                'Touches_Edge': bool(metrics.touches_edge),
                'Dist_To_Edge_um': metrics.dist_to_edge_um,
            })
        lesions_df = pd.DataFrame(lesions_data)

        if summary_metrics:
            summary_data = []
            for i, metrics in enumerate(self.metrics_list, 1):
                has_ablation = metrics.ablation_diameter_um > 0
                summary_data.append({
                    'Lesion_ID': i,
                    'Coagulation_Width_um': round(metrics.coagulation_width_um, 2),
                    'Ablation_Diameter_um': round(metrics.ablation_diameter_um, 2) if has_ablation else 0,
                    'Has_Ablation': has_ablation,
                    'Lesion_Diameter_um': round(metrics.lesion_diameter_um, 2),
                    'Coagulation_Area_um2': round(metrics.coagulation_area_um2, 2),
                    'Ablation_Area_um2': round(metrics.ablation_area_um2, 2) if has_ablation else 0,
                    'Center_X_px': round(metrics.center[0], 1),
                    'Center_Y_px': round(metrics.center[1], 1),
                    'Touches_Edge': bool(metrics.touches_edge),
                    'Dist_To_Edge_um': None if np.isnan(metrics.dist_to_edge_um) else round(metrics.dist_to_edge_um, 2),
                })
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(output_dir, 'lesion_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)

        if len(lesions_data) > 0:
            ablation_areas = lesions_df['Ablation_Area_um2']
            q1, q3 = ablation_areas.quantile(0.25), ablation_areas.quantile(0.75)
            iqr = q3 - q1
            ablation_lower, ablation_upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            coag_areas = lesions_df['Coagulation_Area_um2']
            cq1, cq3 = coag_areas.quantile(0.25), coag_areas.quantile(0.75)
            ciqr = cq3 - cq1
            coag_lower, coag_upper = cq1 - 1.5 * ciqr, cq3 + 1.5 * ciqr
            ablation_filtered = lesions_df[(lesions_df['Ablation_Area_um2'] >= ablation_lower) & (lesions_df['Ablation_Area_um2'] <= ablation_upper)]
            coagulation_filtered = lesions_df[(lesions_df['Coagulation_Area_um2'] >= coag_lower) & (lesions_df['Coagulation_Area_um2'] <= coag_upper)]
            mean_metrics = {
                'Mean_Ablation_Diameter_um': ablation_filtered['Ablation_Diameter_um'].mean(),
                'Mean_Coagulation_Width_um': coagulation_filtered['Coagulation_Width_um'].mean(),
                'Mean_Lesion_Diameter_um': lesions_df['Lesion_Diameter_um'].mean(),
                'Mean_Ablation_Area_um2': ablation_filtered['Ablation_Area_um2'].mean(),
                'Mean_Coagulation_Area_um2': coagulation_filtered['Coagulation_Area_um2'].mean(),
            }
            if len(self.lesion_centers) > 1:
                dist_metrics = self.calculate_adjacent_distances_improved(self.lesion_centers)
                mean_metrics['Mean_Adjacent_Lesion_Distance_um'] = dist_metrics['mean_nearest_neighbor_um']
            mean_df = pd.DataFrame([mean_metrics])
            mean_csv_path = os.path.join(output_dir, 'mean_metrics.csv')
            mean_df.to_csv(mean_csv_path, index=False)

        # 4. Individual lesion crops (after all analytics are computed)
        if figures:
            for i, (metrics, bbox) in enumerate(
                zip(self.metrics_list, filtered_bboxes), 1
            ):
                figure_path = os.path.join(figures_dir, f'lesion_{i:02d}_crop.png')
                self.create_lesion_crop_figure(
                    image, mask, i, metrics, figure_path, bbox)

        return lesions_df

    def create_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_path: str,
    ) -> None:
        """Save mask visualization with lesion IDs and grid visualization."""
        h, w = mask.shape[:2]
        dpi = 200
        scale = min(w, h) / 2000.0

        colored_mask = np.ones((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask == 1] = [0, 255, 0]
        colored_mask[mask == 2] = [0, 0, 255]
        colored_mask[mask == 3] = [255, 0, 0]

        fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax.imshow(colored_mask)
        marker_sz = max(5, 8 * scale)
        font_sz = max(6, 10 * scale)
        offset = max(10, 15 * scale)
        for i, center in enumerate(self.lesion_centers, 1):
            ax.plot(center[0], center[1], 'w+', markersize=marker_sz)
            ax.text(center[0] + offset, center[1] + offset, f'L{i}',
                    color='white', fontsize=font_sz, fontweight='bold')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(output_path, dpi=dpi, pad_inches=0)
        plt.close(fig)

        grid_path = output_path.replace('.png', '_grid.png')
        self._grid_mapper.create_grid_visualization(image, self.lesion_centers, grid_path)
