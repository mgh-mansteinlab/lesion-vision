#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.data.augmentations import build_augmentation_pipeline, build_val_pipeline


class TiledLesionDataset(Dataset):
    """
    Dataset class for loading tiled lesion segmentation data with overlapping shifts.
    Structure must match src.data.tiling: tile_{size}/{shift}/ and mask_{size}/{shift}/,
    with shift in ('no_shift', 'ovlp_20', 'ovlp_40', 'ovlp_60', 'ovlp_80').
    Supports loading from multiple tile sizes (e.g. 448 + 768) simultaneously;
    all tiles are resized to img_size by the augmentation / val pipeline.
    Returns CPU tensors only; never use .cuda() or .to(device) in __getitem__
    to avoid DataLoader workers leaking GPU memory.
    """

    SHIFT_FOLDERS = ('no_shift', 'ovlp_20', 'ovlp_40', 'ovlp_60', 'ovlp_80')

    def __init__(self, data_dir, img_size=(512, 512), shift_type='all',
                 augment=False, is_train=True, n_classes=4,
                 disk_tile_size=None, disk_tile_sizes=None):
        """
        Args:
            data_dir: Root directory containing tile_{size}/ and mask_{size}/ folders.
            img_size: (H, W) model input after resize.
            shift_type: 'all' or one of SHIFT_FOLDERS.
            augment: Apply training augmentations.
            is_train: Training vs validation mode.
            n_classes: Number of segmentation classes.
            disk_tile_size: Single tile size on disk (legacy, e.g. 768).
            disk_tile_sizes: List of tile sizes to load (e.g. [448, 768]).
                             Merged into one dataset. Takes precedence over disk_tile_size.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.shift_type = shift_type
        self.augment = augment
        self.is_train = is_train
        self.n_classes = n_classes

        if disk_tile_sizes is not None:
            sizes = list(disk_tile_sizes)
        elif disk_tile_size is not None:
            sizes = [disk_tile_size]
        else:
            sizes = [img_size[0]]

        shifts_to_use = self.SHIFT_FOLDERS if shift_type == 'all' else (shift_type,)

        self.image_paths = []
        self.mask_paths = []
        for sz in sizes:
            img_root = os.path.join(data_dir, f'tile_{sz}')
            msk_root = os.path.join(data_dir, f'mask_{sz}')
            for shift in shifts_to_use:
                img_glob_path = os.path.join(img_root, shift, '*.tif')
                for img_path in sorted(glob.glob(img_glob_path)):
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = os.path.join(msk_root, shift, base + '.png')
                    if os.path.isfile(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

        if len(self.image_paths) != len(self.mask_paths):
            print(f"Warning: images ({len(self.image_paths)}) != masks ({len(self.mask_paths)})")
        
        # Color mapping for visualization
        self.color_map = {
            'background': [255, 255, 255],  # White
            'tissue': [0, 255, 0],          # Green
            'coagulation': [0, 0, 255],     # Blue
            'ablation': [255, 0, 0]         # Red
        }
        
        # Compute class weights for each sample (for weighted sampling)
        self.sample_weights = None
        self.class_counts = None
        self._aug_pipeline = build_augmentation_pipeline(img_size) if augment else None
        self._val_pipeline = build_val_pipeline(img_size)
        
    def analyze_class_distribution(self, num_samples=None, force_compute=False):
        """
        Analyze the class distribution in the dataset.
        
        Args:
            num_samples: Number of samples to analyze (None = all samples)
            force_compute: Whether to force recomputation of weights
            
        Returns:
            Dictionary with class distribution statistics
        """
        if self.class_counts is not None and not force_compute:
            return self.class_counts
            
        if num_samples is None:
            num_samples = len(self)
        else:
            num_samples = min(num_samples, len(self))
            
        # Randomly select samples to analyze
        indices = np.random.choice(len(self), num_samples, replace=False)
        
        # Count pixels of each class
        class_pixels = np.zeros(self.n_classes)
        total_pixels = 0
        
        # Count samples containing each class
        samples_with_class = np.zeros(self.n_classes)
        
        print(f"Analyzing class distribution across {num_samples} samples...")
        
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                print(f"Processing sample {i}/{num_samples}...")
                
            _, mask = self[idx]
            mask_np = mask.numpy()
            
            # Count pixels
            for c in range(self.n_classes):
                class_count = np.sum(mask_np == c)
                class_pixels[c] += class_count
                if class_count > 0:
                    samples_with_class[c] += 1
            
            total_pixels += mask_np.size
        
        # Calculate statistics
        class_percentages = class_pixels / total_pixels * 100
        sample_percentages = samples_with_class / num_samples * 100
        
        self.class_counts = {
            'class_pixels': class_pixels,
            'total_pixels': total_pixels,
            'pixel_percentages': class_percentages,
            'samples_with_class': samples_with_class,
            'total_samples': num_samples,
            'sample_percentages': sample_percentages
        }
        
        return self.class_counts
    
    def compute_sample_weights(self, alpha=0.75, num_analyze=1000):
        """
        Compute sample weights based on class presence.
        
        Higher weights are assigned to samples containing rare classes.
        
        Args:
            alpha: Weighting factor (0-1) - higher means more emphasis on rare classes
            num_analyze: Number of samples to analyze for computing weights
            
        Returns:
            Numpy array of sample weights
        """
        # Analyze distribution if not already done
        if self.class_counts is None:
            self.analyze_class_distribution(num_samples=num_analyze)
            
        # Compute inverse frequency weights for each class
        sample_percentages = self.class_counts['sample_percentages']
        class_weights = 1.0 / (sample_percentages + 0.01)  # Add small constant to avoid division by zero
        
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights) * self.n_classes
        
        # Compute weights for all samples
        weights = np.ones(len(self))
        
        # We'll analyze a subset of samples to determine their weights
        num_to_process = min(len(self), 1000)
        indices = np.random.choice(len(self), num_to_process, replace=False)
        
        print(f"Computing sample weights for {num_to_process} samples...")
        
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                print(f"Processing sample {i}/{num_to_process}...")
                
            _, mask = self[idx]
            mask_np = mask.numpy()
            
            # Determine which classes are present in this sample
            sample_weight = 1.0
            for c in range(self.n_classes):
                if np.sum(mask_np == c) > 0:
                    # Increase weight based on rare classes (ablation and coagulation)
                    if c >= 2:  # Classes 2 (coagulation) and 3 (ablation) are rare
                        sample_weight *= (1.0 + alpha * class_weights[c])
            
            weights[idx] = sample_weight
            
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        self.sample_weights = weights
        
        return weights
        
    def get_weighted_sampler(self, alpha=0.75, num_analyze=1000):
        """
        Get a weighted sampler for DataLoader to prioritize samples with rare classes.
        
        Args:
            alpha: Weighting factor
            num_analyze: Number of samples to analyze
            
        Returns:
            torch.utils.data.WeightedRandomSampler
        """
        from torch.utils.data import WeightedRandomSampler
        
        if self.sample_weights is None:
            self.compute_sample_weights(alpha, num_analyze)
            
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self),
            replacement=True
        )

    def compute_class_weights(self, num_samples=2000, method='sqrt_inv_freq'):
        """
        Compute per-class loss weights from pixel frequency.

        Args:
            num_samples: Number of tiles to sample (None = all).
            method: 'inv_freq', 'sqrt_inv_freq', or 'median_freq'.

        Returns:
            numpy array of shape (n_classes,) with per-class weights.
        """
        dist = self.analyze_class_distribution(num_samples=num_samples)
        counts = dist['class_pixels']
        total = dist['total_pixels']
        freq = counts / total

        if method == 'inv_freq':
            weights = total / (self.n_classes * counts + 1e-6)
        elif method == 'sqrt_inv_freq':
            weights = np.sqrt(total / (self.n_classes * counts + 1e-6))
        elif method == 'median_freq':
            median_f = np.median(freq)
            weights = median_f / (freq + 1e-6)
        else:
            raise ValueError(f"Unknown method: {method}")

        weights = weights / weights.min()
        print(f"Class weights ({method}): {dict(zip(['BG','Tissue','Coag','Ablation'], weights))}")
        return weights.astype(np.float32)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image-mask pair from the dataset.
        Returns CPU tensors only; do not use .cuda() or .to(device) here
        to avoid DataLoader workers leaking GPU memory.
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask as uint8 RGB numpy arrays
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Convert mask RGB to class indices before augmentation
        h, w = mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        white_mask = (mask[..., 0] > 240) & (mask[..., 1] > 240) & (mask[..., 2] > 240)
        class_mask[white_mask] = 0
        green_mask = (mask[..., 1] > 200) & (mask[..., 0] < 100) & (mask[..., 2] < 100)
        class_mask[green_mask] = 1
        blue_mask = (mask[..., 2] > 200) & (mask[..., 0] < 100) & (mask[..., 1] < 100)
        class_mask[blue_mask] = 2
        red_mask = (mask[..., 0] > 200) & (mask[..., 1] < 100) & (mask[..., 2] < 100)
        class_mask[red_mask] = 3

        # Apply augmentation (train) or just resize (val)
        if self.augment and self.is_train and self._aug_pipeline is not None:
            transformed = self._aug_pipeline(image=image, mask=class_mask)
        else:
            transformed = self._val_pipeline(image=image, mask=class_mask)
        
        image = transformed['image']
        class_mask = transformed['mask']
        
        # Convert to tensors
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(class_mask.astype(np.int64)).long()
        
        return img_tensor, mask_tensor


def get_train_val_dataset(
    data_dir,
    img_size=(512, 512),
    shift_type='all',
    augment=True,
    val_split=0.2,
    random_state=42,
    disk_tile_size=None,
    disk_tile_sizes=None,
):
    """
    Split the dataset into training and validation sets.

    Args:
        data_dir: Root directory containing tile_{size}/ and mask_{size}/ folders.
        img_size: (H, W) model input after resize.
        shift_type: 'all' or a specific shift folder name.
        augment: Apply training augmentations.
        val_split: Fraction held out for validation.
        random_state: Seed for reproducible splits.
        disk_tile_size: Single tile size on disk (legacy).
        disk_tile_sizes: List of tile sizes (e.g. [448, 768]). Takes precedence.

    Returns:
        (train_dataset, val_dataset)
    """
    ds_kwargs = dict(
        data_dir=data_dir,
        img_size=img_size,
        shift_type=shift_type,
        disk_tile_size=disk_tile_size,
        disk_tile_sizes=disk_tile_sizes,
    )

    full_dataset = TiledLesionDataset(**ds_kwargs, augment=False, is_train=True)

    n_samples = len(full_dataset.image_paths)
    if n_samples == 0:
        sizes = disk_tile_sizes or ([disk_tile_size] if disk_tile_size else [img_size[0]])
        dirs = ", ".join(f"tile_{s}" for s in sizes)
        shifts = "no_shift, ovlp_20, ovlp_40, ovlp_60, ovlp_80" if shift_type == "all" else shift_type
        raise ValueError(
            f"No image/mask pairs found (n_samples=0). "
            f"Expected tiled data under {dirs} in {data_dir!r} "
            f"with shift folders: {shifts}. "
            f"Run src.data.tiling to generate tiles."
        )

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        full_dataset.image_paths,
        full_dataset.mask_paths,
        test_size=val_split,
        random_state=random_state,
    )

    train_dataset = TiledLesionDataset(**ds_kwargs, augment=augment, is_train=True)
    train_dataset.image_paths = train_imgs
    train_dataset.mask_paths = train_masks

    val_dataset = TiledLesionDataset(**ds_kwargs, augment=False, is_train=False)
    val_dataset.image_paths = val_imgs
    val_dataset.mask_paths = val_masks

    return train_dataset, val_dataset
