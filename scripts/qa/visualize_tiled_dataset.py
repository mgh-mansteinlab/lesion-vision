#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import TiledLesionDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize tiled dataset samples')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing tile and mask directories')
    parser.add_argument('--output_dir', type=str, default='tiled_visualization_results',
                        help='Directory to save visualizations')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size (square)')
    parser.add_argument('--num_samples', type=int, default=40,
                        help='Number of samples to visualize')
    parser.add_argument('--min_ablation', type=int, default=0,
                        help='Minimum number of ablation samples to include')
    parser.add_argument('--min_coagulation', type=int, default=0,
                        help='Minimum number of coagulation samples to include')
    parser.add_argument('--shift_type', type=str, default='all', 
                        choices=['all', 'no_shift', 'ovlp_20', 'ovlp_40', 'ovlp_60', 'ovlp_80'],
                        help='Type of shift/overlap to use (must match TileGen)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def find_samples_with_class(dataset, class_idx, min_count=10, max_samples=1000):
    """
    Find samples that contain a specific class.
    
    Args:
        dataset: TiledLesionDataset instance
        class_idx: Class index to search for (0=background, 1=tissue, 2=coagulation, 3=ablation)
        min_count: Minimum number of samples to find
        max_samples: Maximum number of samples to examine
        
    Returns:
        List of indices of samples containing the specified class
    """
    indices_with_class = []
    
    # Use a subset of the dataset if it's large
    total_to_check = min(len(dataset), max_samples)
    indices_to_check = np.random.choice(len(dataset), total_to_check, replace=False)
    
    print(f"Searching for samples with class {class_idx} (examining {total_to_check} samples)...")
    
    for i, idx in enumerate(indices_to_check):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{total_to_check} samples, found {len(indices_with_class)} with class {class_idx}")
            
        _, mask = dataset[idx]
        mask_np = mask.numpy()
        
        # Check if class is present
        if np.sum(mask_np == class_idx) > 100:  # At least 100 pixels to be significant
            indices_with_class.append(idx)
            
        # Early exit if we found enough samples
        if len(indices_with_class) >= min_count:
            break
            
    print(f"Found {len(indices_with_class)} samples containing class {class_idx}")
    return indices_with_class


def visualize_samples_with_filenames(dataset, indices, save_path, figsize=(20, 20)):
    """
    Visualize specific samples from the dataset with their filenames.
    
    Args:
        dataset: TiledLesionDataset instance
        indices: List of indices to visualize
        save_path: Path to save the visualization
        figsize: Figure size (width, height) in inches
    """
    num_samples = len(indices)
    
    # Determine grid dimensions based on number of samples
    grid_size = math.ceil(math.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=figsize)
    
    # Define class colors to ensure consistency
    class_colors = [
        [1.0, 1.0, 1.0],  # 0: Background (white)
        [0.0, 1.0, 0.0],  # 1: Tissue (green)
        [0.0, 0.0, 1.0],  # 2: Coagulation (blue)
        [1.0, 0.0, 0.0],  # 3: Ablation (red)
    ]
    
    # Create a flattened view of axes for easy indexing
    axes_flat = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= num_samples:
            break
            
        # Get image and mask
        image, mask = dataset[idx]
        
        # Get filename
        image_path = dataset.image_paths[idx]
        filename = os.path.basename(image_path)
        
        # Convert to numpy
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        
        # Calculate class percentages for this sample
        class_counts = []
        for c in range(4):  # Assuming 4 classes
            count = np.sum(mask_np == c)
            percentage = count / mask_np.size * 100
            class_counts.append(f"{percentage:.1f}%")
            
        # Create color mask
        color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.float32)
        for cls_idx, color in enumerate(class_colors):
            color_mask[mask_np == cls_idx] = color
        
        # Display image with filename
        ax_img = axes_flat[i * 2]
        ax_img.imshow(image_np)
        ax_img.set_title(f"{filename}", fontsize=8)
        ax_img.axis('off')
        
        # Display mask with class percentages
        ax_mask = axes_flat[i * 2 + 1]
        ax_mask.imshow(color_mask)
        
        # Show class percentages in the title
        class_text = f"BG: {class_counts[0]}, Tis: {class_counts[1]}, Coa: {class_counts[2]}, Abl: {class_counts[3]}"
        ax_mask.set_title(class_text, fontsize=8)
        ax_mask.axis('off')
    
    # Hide any unused subplots
    for i in range(2 * num_samples, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Add a legend for mask colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=class_colors[0], label='Background'),
        plt.Rectangle((0, 0), 1, 1, color=class_colors[1], label='Tissue'),
        plt.Rectangle((0, 0), 1, 1, color=class_colors[2], label='Coagulation'),
        plt.Rectangle((0, 0), 1, 1, color=class_colors[3], label='Ablation')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    dataset = TiledLesionDataset(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        shift_type=args.shift_type,
        augment=False,
        is_train=True
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"Error: No image-mask pairs found in {args.data_dir} with shift_type={args.shift_type}")
        return
    
    print(f"\nFound {len(dataset)} image-mask pairs.")
    
    # Analyze class distribution
    class_stats = dataset.analyze_class_distribution(num_samples=1000)
    print("\nClass distribution (% of samples containing each class):")
    for i, percent in enumerate(class_stats['sample_percentages']):
        class_name = ['Background', 'Tissue', 'Coagulation', 'Ablation'][i]
        print(f"  {class_name}: {percent:.2f}% of samples")
    
    # Find samples with ablation
    if args.min_ablation > 0:
        ablation_indices = find_samples_with_class(dataset, 3, min_count=args.min_ablation, max_samples=5000)
        
        # Visualize ablation samples
        if ablation_indices:
            ablation_path = os.path.join(args.output_dir, "samples_with_ablation.png")
            visualize_samples_with_filenames(dataset, ablation_indices, ablation_path, figsize=(30, 30))
            print(f"Ablation samples visualization saved to {ablation_path}")
    
    # Find samples with coagulation
    if args.min_coagulation > 0:
        coagulation_indices = find_samples_with_class(dataset, 2, min_count=args.min_coagulation, max_samples=5000)
        
        # Visualize coagulation samples
        if coagulation_indices:
            coagulation_path = os.path.join(args.output_dir, "samples_with_coagulation.png")
            visualize_samples_with_filenames(dataset, coagulation_indices, coagulation_path, figsize=(30, 30))
            print(f"Coagulation samples visualization saved to {coagulation_path}")
    
    # Generate a balanced set of samples
    print(f"\nCreating balanced visualization with {args.num_samples} samples...")
    balanced_indices = []
    
    # Include ablation samples (25% of total)
    ablation_count = args.num_samples // 4
    if ablation_count > 0:
        ablation_indices = find_samples_with_class(dataset, 3, min_count=ablation_count, max_samples=10000)
        balanced_indices.extend(ablation_indices[:ablation_count])
    
    # Include coagulation samples (25% of total)
    coagulation_count = args.num_samples // 4
    if coagulation_count > 0:
        coagulation_indices = find_samples_with_class(dataset, 2, min_count=coagulation_count, max_samples=10000)
        # Filter out indices already included from ablation
        coagulation_indices = [idx for idx in coagulation_indices if idx not in balanced_indices]
        balanced_indices.extend(coagulation_indices[:coagulation_count])
    
    # Fill remaining slots with random samples
    remaining_count = args.num_samples - len(balanced_indices)
    if remaining_count > 0:
        all_indices = set(range(len(dataset)))
        remaining_indices = list(all_indices - set(balanced_indices))
        selected_indices = np.random.choice(remaining_indices, min(remaining_count, len(remaining_indices)), replace=False)
        balanced_indices.extend(selected_indices)
    
    # Visualize balanced samples
    balanced_path = os.path.join(args.output_dir, "balanced_samples.png")
    visualize_samples_with_filenames(dataset, balanced_indices, balanced_path, figsize=(30, 30))
    print(f"Balanced visualization saved to {balanced_path}")
    
    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
