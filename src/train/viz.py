import matplotlib.pyplot as plt
import numpy as np
import torch


def create_color_mask(mask, class_colors=None):
    """
    Create a color-coded mask for visualization.
    
    Args:
        mask: Segmentation mask with class indices (numpy array or tensor)
        class_colors: List of colors for each class [background, tissue, coagulation, ablation]
                     If None, uses default colors: [white, green, blue, red]
    
    Returns:
        Color-coded mask (RGB format)
    """
    if class_colors is None:
        # Default colors: white=background, green=tissue, blue=coagulation, red=ablation
        from src.constants import CLASS_COLORS_FLOAT
        class_colors = CLASS_COLORS_FLOAT
    
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Create empty RGB image
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)
    
    # Fill RGB channels based on class indices
    for i, color in enumerate(class_colors):
        # Create a boolean mask where the class index matches
        indices = mask == i
        # Apply the color to those pixels
        rgb_mask[indices] = color
    
    # Safety check - ensure any unlabeled pixels are visible (assign white)
    unlabeled = (mask < 0) | (mask >= len(class_colors))
    if np.any(unlabeled):
        rgb_mask[unlabeled] = [1.0, 1.0, 1.0]  # White for any unlabeled pixels
    
    return rgb_mask


def visualize_results(image, gt_mask, pred_mask, title=None):
    """
    Visualize image, ground truth mask, and predicted mask.
    
    Args:
        image: Input image (numpy array [H,W,C])
        gt_mask: Ground truth mask (numpy array [H,W,3])
        pred_mask: Predicted mask (numpy array [H,W,3])
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]
    
    # Denormalize image if needed (if values are in range [0,1])
    if image.max() <= 1.0:
        image = image * 255.0
    
    # Ensure image is in range [0, 255] and uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display ground truth mask
    axes[1].imshow(gt_mask)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Display predicted mask
    axes[2].imshow(pred_mask)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    return fig


def plot_training_metrics(history, save_path=None):
    """
    Plot training metrics.
    
    Args:
        history: Dictionary containing training history (loss and accuracy)
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig
