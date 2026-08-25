import os
# Set OpenCV to allow large images (for NDPI processing) - must be before cv2 import
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

import cv2
import gc
import glob
import json
import shutil
import traceback
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.analytics.lesions import LesionAnalyzer, LesionMetrics
from src.constants import CLASS_COLORS_RGB, colorize_class_mask
from src.infer.image_io import read_rgb, resolve_ndpi
from src.infer.postprocess import post_process_mask
from src.infer.tiles import (
    TissueWindowConfig,
    get_tiles,
    get_tissue_bbox,
    is_background_tile,
    stitch_predictions,
    tissue_threshold,
)
from src.models.segmentation_model import LesionSegmentationModel

# Re-export for backward compatibility (e.g. multi_gpu_batch_predictor)
__all__ = ['LesionPredictor', 'LesionMetrics', 'LesionAnalyzer']


def _default_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class LesionPredictor:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the lesion predictor.
        
        Args:
            model_path: Path to the trained model. If None, uses
                models/single_scale_448/best_model.pth when present, otherwise
                the lexicographically last models/*/best_model.pth.
            device: Device to run inference on ('cuda', 'mps', or 'cpu').
                    Auto-detected if not provided (cuda > mps > cpu).
        """
        self.device = device or _default_device()
        print(f"Using device: {self.device}")
        self.analyzer = LesionAnalyzer()
        
        # Find model path if not provided
        if model_path is None:
            preferred = os.path.join('models', 'single_scale_448', 'best_model.pth')
            if os.path.isfile(preferred):
                model_path = preferred
                print(f"Using default checkpoint: {model_path}")
            else:
                model_dirs = [
                    d for d in sorted(glob.glob('models/*'))
                    if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'best_model.pth'))
                ]
                if not model_dirs:
                    raise ValueError(
                        "No model checkpoints found under models/*/best_model.pth. "
                        "Weights are provided on request (see models/DATA_ON_REQUEST.txt)."
                    )
                latest_model_dir = model_dirs[-1]
                model_path = os.path.join(latest_model_dir, 'best_model.pth')
                print(f"Using checkpoint from: {model_path}")
        
        # Load model configuration
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded model configuration: {config}")
        else:
            config = {
                'n_classes': 4,
                'img_size': 512,
                'learning_rate': 1e-4,
            }
            print("Warning: No config.json found, using default configuration")
        
        img_size = config.get('img_size', 512)
        self.img_size = img_size
        
        # Create model with same configuration as training
        total_epochs = config.get('epochs', 100)
        self.model = LesionSegmentationModel(
            n_classes=config.get('n_classes', 4),
            backbone=config.get('backbone', 'tu-convnext_base'),
            encoder_weights=config.get('encoder_weights', 'imagenet'),
            learning_rate=config.get('learning_rate', 1e-4),
            input_shape=(3, img_size, img_size),
            device=self.device,
            use_amp=False,
            gradient_accumulation_steps=1,
            activation_checkpointing=config.get('activation_checkpointing', False),
            loss_type=config.get('loss_type', 'compound'),
            class_weights=None,
            scheduler_type=config.get('scheduler_type', 'cosine'),
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=total_epochs,
        )
        
        # Load trained weights
        self.model.load(model_path, device=self.device)
        
        if hasattr(self.model, 'model'):
            self.model.model = self.model.model.to(self.device)
            self.model.model.eval()
        else:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # Color mapping for visualization
        self.color_map = dict(CLASS_COLORS_RGB)
        self.tissue_cfg = TissueWindowConfig()

        # --- Tissue-region / background-tile detection ---------------------
        # Strict bounding box around the main sample (only a little padding),
        # plus background-tile skipping so blank slide area is never sent to the
        # model and is directly assigned background (white).
        #
        # The slide background is a bright, near-white mode (~gray 234-238) while
        # tissue — even PALE tissue — sits below it.  We therefore separate the
        # two with a threshold a fixed margin below the estimated background
        # brightness (a high percentile of the image).  This single threshold is
        # shared by the bbox detector and the tile-skip test so pale tissue is
        # never cropped out or skipped.
        self.tissue_padding = self.tissue_cfg.padding
        self.tissue_bg_percentile = self.tissue_cfg.bg_percentile
        self.tissue_bg_margin = self.tissue_cfg.bg_margin
        self.tissue_min_fg_ratio = self.tissue_cfg.min_fg_ratio


    def _tissue_threshold(self, gray: np.ndarray) -> float:
        return tissue_threshold(gray, self.tissue_cfg)

    def _is_background_tile(self, tile: np.ndarray, threshold: float) -> bool:
        return is_background_tile(tile, threshold, self.tissue_cfg)

    def get_tissue_bbox(
        self,
        image: np.ndarray,
        padding: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        return get_tissue_bbox(image, self.tissue_cfg, padding=padding, threshold=threshold)

    def get_tiles(self, image: np.ndarray, tile_size: int = 512, overlap: int = 128):
        return get_tiles(image, tile_size=tile_size, overlap=overlap)

    def stitch_predictions(self, tiles, original_shape, tile_size=512, overlap=128):
        return stitch_predictions(tiles, original_shape, tile_size=tile_size, overlap=overlap)

    def predict_tile(self, tile: np.ndarray, return_probs: bool = True) -> np.ndarray:
        """
        Make prediction for a single tile.
        
        Args:
            tile: Input tile (H, W, 3) uint8.
            return_probs: If True, return (n_classes, H, W) softmax probabilities.
                          If False, return (H, W) class indices (legacy behaviour).
        """
        tile_f = tile.astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile_f).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            model = self.model.model if hasattr(self.model, 'model') else self.model
            logits = model(tensor)
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                return probs.squeeze(0).cpu().numpy()
            return torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    
    def predict_tile_tta(self, tile: np.ndarray) -> np.ndarray:
        """
        Predict a single tile with test-time augmentation.
        Averages softmax probabilities over flips and 90-degree rotations.
        Returns (n_classes, H, W) probability map.
        """
        accum = None
        n = 0
        for k in range(4):
            rotated = np.rot90(tile, k).copy()
            for flip_code in (None, 1):
                aug = cv2.flip(rotated, flip_code) if flip_code is not None else rotated
                probs = self.predict_tile(aug, return_probs=True)
                if flip_code is not None:
                    probs = probs[:, :, ::-1].copy()
                probs = np.rot90(probs, -k, axes=(1, 2)).copy()
                if accum is None:
                    accum = probs
                else:
                    accum += probs
                n += 1
        return accum / n

    def _read_image(self, image_path: str) -> np.ndarray:
        return read_rgb(image_path)

    def _resolve_ndpi(self, image_path: str) -> Tuple[str, Optional[str]]:
        return resolve_ndpi(image_path)

    def _predict_image(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int = 128,
        use_tta: bool = False,
        use_post_process: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Core prediction on a loaded RGB image.

        Returns:
            (cropped_prediction_mask, bbox) where bbox is (x1, y1, x2, y2) in
            the original image coordinate space.
        """
        # One background-relative threshold for both the bbox and tile skipping,
        # estimated from the full image where ample background is present.
        gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = self._tissue_threshold(cv2.GaussianBlur(gray_full, (5, 5), 0))

        cropped_image, bbox = self.get_tissue_bbox(image, threshold=threshold)

        tiles = self.get_tiles(cropped_image, tile_size, overlap)

        n_classes = len(self.color_map)
        # Reusable background probability map (class 0 = background, prob 1.0)
        bg_prob = np.zeros((n_classes, tile_size, tile_size), dtype=np.float32)
        bg_prob[0] = 1.0

        predictions = []
        n_skipped = 0
        tile_fn = self.predict_tile_tta if use_tta else self.predict_tile
        for tile, pos in tqdm(tiles, desc="Processing tiles", unit="tile"):
            # Blank-slide tiles (essentially no tissue pixels) are assigned white
            # directly and skipped.  Any tile with partial/full sample area is
            # sent to the model.
            if self._is_background_tile(tile, threshold):
                predictions.append((bg_prob, pos))
                n_skipped += 1
                continue
            pred = tile_fn(tile)
            predictions.append((pred, pos))

        if n_skipped:
            print(f"Skipped {n_skipped}/{len(tiles)} near-white background tiles")

        stitched = self.stitch_predictions(predictions, cropped_image.shape[:2], tile_size, overlap)
        if use_post_process:
            stitched = self.post_process_mask(stitched)
        return stitched, bbox

    def predict(
        self,
        image_path: str,
        tile_size: int = 512,
        overlap: int = 128,
        use_tta: bool = False,
        use_post_process: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Make prediction for an image file using sliding window approach.

        Args:
            image_path: Path to the input image (supports .tif, .ndpi)
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            Tuple of (predicted segmentation mask, (x1, y1, x2, y2))
        """
        resolved_path, temp_dir = self._resolve_ndpi(image_path)
        try:
            image = self._read_image(resolved_path)
            return self._predict_image(
                image, tile_size, overlap, use_tta=use_tta, use_post_process=use_post_process
            )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        return colorize_class_mask(mask, self.color_map)

    # ------------------------------------------------------------------
    # Post-processing: enforce spatial constraints
    # ------------------------------------------------------------------
    # Class indices: 0=Background, 1=Tissue, 2=Coagulation, 3=Ablation
    #
    # Domain rules:
    #   1. White/tissue open areas (cracks, gaps) may exist inside the main
    #      sample and are preserved.  Only regions that are FULLY ENCLOSED by
    #      lesion (coag/ablation) are reassigned — to the nearest lesion class.
    #   2. Ablation (if present) is always surrounded by coagulation — it
    #      must not directly touch tissue or background.
    # ------------------------------------------------------------------

    @staticmethod
    def post_process_mask(mask: np.ndarray) -> np.ndarray:
        return post_process_mask(mask)

    def save_prediction(self, image_path: str, output_path: str, tile_size: int = 512,
                        overlap: int = 128, summary_metrics: bool = False,
                        figures: bool = False, create_subdir: bool = True,
                        use_tta: bool = False, use_post_process: bool = True):
        """
        Make prediction and save the result as a colored mask and overlay.

        Outputs are organised into subdirectories inside the per-image folder::

            <image_name>/
                images/              masks, overlays, visualisation PNGs
                csv/                 metric CSV files
                individual_lesions/  per-lesion crop figures

        Args:
            image_path: Path to the input image
            output_path: Used to derive the parent output directory.
            tile_size: Size of each tile
            overlap: Overlap between tiles
            summary_metrics: Whether to create summary CSV with lesion metrics
            figures: Whether to create lesion crop figures with analytics
            create_subdir: Whether to create a per-image subdirectory (default True)
        """
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if create_subdir:
            output_dir = os.path.abspath(os.path.dirname(output_path))
            image_dir = os.path.join(output_dir, image_name)
        else:
            image_dir = os.path.abspath(os.path.dirname(output_path))

        images_dir = os.path.join(image_dir, 'images')
        csv_dir = os.path.join(image_dir, 'csv')
        lesions_dir = os.path.join(image_dir, 'individual_lesions')
        for d in [images_dir, csv_dir, lesions_dir]:
            os.makedirs(d, exist_ok=True)

        resolved_path, temp_dir = self._resolve_ndpi(image_path)
        try:
            original = self._read_image(resolved_path)
            pred, bbox = self._predict_image(
                original, tile_size, overlap, use_tta=use_tta, use_post_process=use_post_process
            )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        full_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        full_mask[y1:y2, x1:x2] = pred

        colored_mask = self._colorize_mask(full_mask)
        overlay = cv2.addWeighted(original, 0.7, colored_mask, 0.3, 0)

        mask_path = os.path.join(images_dir, f"{image_name}_pred.png")
        cv2.imwrite(mask_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
        overlay_path = os.path.join(images_dir, f"{image_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved prediction mask to: {mask_path}")
        print(f"Saved overlay to: {overlay_path}")

        if summary_metrics or figures:
            self._run_analysis(original, full_mask, image_name,
                               images_dir, csv_dir, lesions_dir)
        else:
            print("Skipping lesion analysis (no analysis flags enabled)")

        self._save_triptych(original, colored_mask, overlay, images_dir, image_name)
        gc.collect()

    def _run_analysis(self, original: np.ndarray, full_mask: np.ndarray,
                      image_name: str, images_dir: str, csv_dir: str,
                      individual_lesions_dir: str) -> None:
        """Run lesion analysis, corrected-mask outputs, and radial mushrooming."""
        print("Starting lesion analysis...")
        try:
            self.analyzer.analyze_mask(
                full_mask, original, csv_dir, True, True,
                figures_dir=individual_lesions_dir,
            )
            print("Lesion analysis completed successfully")
        except Exception as e:
            print(f"Error in lesion analysis: {e}")
            traceback.print_exc()

        vis_path = os.path.join(images_dir, f"{image_name}_analysis.png")
        self.analyzer.create_visualization(original, full_mask, vis_path)
        print(f"Saved analysis visualization to: {vis_path}")

        if len(self.analyzer.lesion_centers) >= 2 and len(self.analyzer.metrics_list) >= 2:
            try:
                from src.analytics.radial import RadialMushroomingAnalyzer
                rma = RadialMushroomingAnalyzer(n_radial_bins=10)
                df_radial = rma.analyze(
                    self.analyzer.lesion_centers,
                    self.analyzer.metrics_list,
                    mask_shape=full_mask.shape,
                )
                if not df_radial.empty:
                    radial_csv = os.path.join(csv_dir, f"{image_name}_radial_cv_analysis.csv")
                    df_radial.to_csv(radial_csv, index=False)
                    print(f"Saved radial CV analysis to: {radial_csv}")

                    radial_plot = os.path.join(images_dir, f"{image_name}_radial_cv_plot.png")
                    rma.plot_cv_vs_radius(df_radial, radial_plot)
                    print(f"Saved radial CV plot to: {radial_plot}")

                    diagram_path = os.path.join(images_dir, f"{image_name}_radial_mushrooming_diagram.png")
                    rma.plot_radial_mushrooming_diagram(
                        original, full_mask,
                        self.analyzer.lesion_centers, self.analyzer.metrics_list,
                        df_radial, diagram_path,
                    )
                    print(f"Saved radial mushrooming diagram to: {diagram_path}")

                    detailed_path = os.path.join(images_dir, f"{image_name}_radial_detailed_analytics.png")
                    rma.plot_detailed_analytics(df_radial, detailed_path)
                    print(f"Saved radial detailed analytics to: {detailed_path}")
            except Exception as e:
                print(f"Radial mushrooming analysis skipped: {e}")

    def _save_triptych(self, original: np.ndarray, colored_mask: np.ndarray,
                       overlay: np.ndarray, image_dir: str, image_name: str) -> None:
        """Save the 3-panel visualization (original | mask | overlay)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(colored_mask)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array(color) / 255, label=label)
            for label, color in {
                'Background': self.color_map[0],
                'Tissue': self.color_map[1],
                'Coagulation': self.color_map[2],
                'Ablation': self.color_map[3],
            }.items()
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4)

        vis_path = os.path.join(image_dir, f"{image_name}_visualization.png")
        plt.tight_layout()
        plt.savefig(vis_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved visualization to: {vis_path}")

    def process_directory(self, input_dir: str, output_dir: str, tile_size: int = 512,
                          overlap: int = 128, summary_metrics: bool = False,
                          figures: bool = False, use_tta: bool = False,
                          use_post_process: bool = True):
        """
        Process all .tif/.ndpi files in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save predictions
            tile_size: Size of each tile
            overlap: Overlap between tiles
            summary_metrics: Whether to create summary CSV with lesion metrics
            figures: Whether to create lesion crop figures with analytics
        """
        os.makedirs(output_dir, exist_ok=True)

        image_paths = glob.glob(os.path.join(input_dir, '*.tif'))
        image_paths.extend(glob.glob(os.path.join(input_dir, '*.ndpi')))

        for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
            filename = os.path.basename(image_path)
            base = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base}_pred.png")
            self.save_prediction(
                image_path, output_path, tile_size, overlap,
                summary_metrics=summary_metrics, figures=figures,
                use_tta=use_tta, use_post_process=use_post_process,
            )