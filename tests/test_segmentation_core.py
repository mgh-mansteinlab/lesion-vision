"""Unit tests for class map, tiling stitch, and post-process."""

import numpy as np

from src.constants import CLASS_NAMES
from src.eval.metrics import rgb_to_class_mask
from src.infer.postprocess import post_process_mask
from src.infer.tiles import TissueWindowConfig, get_tiles, get_tissue_bbox, stitch_predictions
from src.models.architecture import SMPSegmentationNet, uses_plain_unet
from src.models.segmentation_model import LesionSegmentationModel


def test_class_names_match_readme_convention():
    assert CLASS_NAMES == {
        0: "Background",
        1: "Tissue",
        2: "Coagulation",
        3: "Ablation",
    }
    model = LesionSegmentationModel(
        encoder_weights=None,
        device="cpu",
        use_amp=False,
        activation_checkpointing=False,
    )
    assert model.class_names == CLASS_NAMES


def test_default_convnext_uses_plain_unet_not_unetplusplus():
    assert uses_plain_unet("tu-convnext_base") is True
    assert uses_plain_unet("tu-convnext_small") is True
    assert uses_plain_unet("efficientnet-b3") is False
    assert uses_plain_unet("resnet34") is False
    net = SMPSegmentationNet(n_classes=4, backbone="tu-convnext_base", encoder_weights=None)
    assert type(net.model).__name__ == "Unet"


def test_rgb_to_class_mask_colors():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [255, 255, 255]
    img[0, 1] = [0, 255, 0]
    img[1, 0] = [0, 0, 255]
    img[1, 1] = [255, 0, 0]
    cls = rgb_to_class_mask(img)
    assert cls[0, 0] == 0
    assert cls[0, 1] == 1
    assert cls[1, 0] == 2
    assert cls[1, 1] == 3


def test_stitch_probability_argmax_recovers_constant_class():
    tile_size = 32
    overlap = 0
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    tiles = get_tiles(image, tile_size=tile_size, overlap=overlap)
    n_classes = 4
    preds = []
    for _tile, pos in tiles:
        prob = np.zeros((n_classes, tile_size, tile_size), dtype=np.float32)
        prob[1] = 1.0
        preds.append((prob, pos))
    stitched = stitch_predictions(preds, image.shape[:2], tile_size=tile_size, overlap=overlap)
    assert stitched.shape == (40, 40)
    assert np.all(stitched == 1)


def test_tissue_bbox_keeps_pale_punch_not_just_dark_blob():
    """Pale NBTC must not collapse to the darkest stain island."""
    h, w = 240, 240
    image = np.full((h, w, 3), 238, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    punch = (xx - 120) ** 2 + (yy - 120) ** 2 <= 85 ** 2
    image[punch] = 228
    image[100:130, 100:130] = 90
    cfg = TissueWindowConfig(padding=4, min_span_px=100, bg_percentile=98, bg_margin=4)
    crop, bbox = get_tissue_bbox(image, cfg)
    x1, y1, x2, y2 = bbox
    assert x2 - x1 >= 150
    assert y2 - y1 >= 150
    assert x1 <= 40 and y1 <= 40
    assert crop.shape[0] >= 150 and crop.shape[1] >= 150


def test_post_process_converts_exposed_ablation_to_coagulation():
    mask = np.ones((20, 20), dtype=np.uint8)  # tissue
    mask[6:14, 6:14] = 2  # coagulation ring
    mask[8:12, 8:12] = 3  # ablation core
    mask[8, 6] = 3  # ablation pixel touching tissue
    out = post_process_mask(mask)
    assert out[8, 6] == 2
    assert np.any(out == 3)
