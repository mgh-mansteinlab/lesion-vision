"""Albumentations pipelines for tiled training data."""

import cv2
import albumentations as A


def build_augmentation_pipeline(img_size=(512, 512)):
    """
    Build an albumentations augmentation pipeline for training.
    Operates on numpy HWC uint8 images and HW uint8 masks.

    Lesion-focused additions (beyond the original geometric/photometric set):
    - RandomSizedBBox-free crop variants are not used because tiles are
      already extracted at two scales; instead we add stain-robustness
      transforms (HueSaturationValue, RandomBrightnessContrast, sharpen)
      that vary the NBTC/coagulation contrast the model must generalize
      across, and Downscale/Upscale to simulate focus variation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.85, 1.15), rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT, fill=255,
            fill_mask=0, p=0.5,
        ),
        A.ElasticTransform(
            alpha=80, sigma=80 * 0.05,
            border_mode=cv2.BORDER_CONSTANT, fill=255,
            fill_mask=0, p=0.3,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.Downscale(scale_range=(0.5, 0.9), p=1.0),
        ], p=0.25),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.OneOf([
            A.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.15, hue=0.05, p=1.0,
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                                  val_shift_limit=25, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.3, p=1.0),
        ], p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2), p=0.2),
        A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_AREA),
    ])


def build_val_pipeline(img_size=(512, 512)):
    """Validation/inference pipeline: resize only."""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_AREA),
    ])
