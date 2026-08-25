"""Shared class ids, display names, colors, and physical scale."""

from typing import Dict, List, Tuple

CLASS_ID_BACKGROUND = 0
CLASS_ID_TISSUE = 1
CLASS_ID_COAGULATION = 2
CLASS_ID_ABLATION = 3

CLASS_NAMES: Dict[int, str] = {
    CLASS_ID_BACKGROUND: "Background",
    CLASS_ID_TISSUE: "Tissue",
    CLASS_ID_COAGULATION: "Coagulation",
    CLASS_ID_ABLATION: "Ablation",
}

CLASS_NAME_LIST: List[str] = [
    CLASS_NAMES[i] for i in range(len(CLASS_NAMES))
]

# RGB uint8 used in saved masks and overlays.
CLASS_COLORS_RGB = {
    CLASS_ID_BACKGROUND: [255, 255, 255],
    CLASS_ID_TISSUE: [0, 255, 0],
    CLASS_ID_COAGULATION: [0, 0, 255],
    CLASS_ID_ABLATION: [255, 0, 0],
}

# Matplotlib 0-1 colors for training visualizations.
CLASS_COLORS_FLOAT: List[List[float]] = [
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
]

# Pixel-to-micrometer scale at OpenSlide level 2.
PIXEL_SCALE_UM = 0.221 * (2 ** 2)

# Manuscript definition of edge-adjacent / peripheral lesions.
PERIPHERAL_EDGE_UM = 500.0

N_CLASSES = 4


def colorize_class_mask(mask, color_map: Dict[int, List[int]] | None = None):
    """Convert a class-index mask to an RGB uint8 image."""
    import numpy as np

    colors = color_map or CLASS_COLORS_RGB
    colored = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for class_idx, color in colors.items():
        colored[mask == class_idx] = color
    return colored
