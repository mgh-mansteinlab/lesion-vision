"""Training CLI."""

import os
import sys

import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from src.train.distributed import cleanup
from src.train.loop import train_process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing tile and mask directories')
    parser.add_argument('--output_dir', type=str, default='model_output',
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--backbone', type=str, default='tu-convnext_base',
                        help='Encoder backbone (e.g. tu-convnext_base, tu-convnext_small, efficientnet-b3)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size for training (square)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU (keep small for L4 24GB; use --gradient_accumulation_steps for effective batch)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective_batch = batch_size * accumulation_steps * world_size)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision (autocast + GradScaler); set --no_amp to disable')
    parser.add_argument('--no_amp', action='store_false', dest='amp',
                        help='Disable mixed precision')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='Number of segmentation classes')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--split_level', type=str, default='tile',
                        choices=['tile', 'slide', 'punch'],
                        help="Split unit: 'tile' (legacy; can leak slides between subsets), "
                             "'slide' (whole-slide-disjoint split; all tiles of a slide "
                             "stay in the same subset), or 'punch' (punch-disjoint; all "
                             "slides of a punch stay in the same subset; the punch is the "
                             "analysis unit and the closest proxy for the donor)")
    parser.add_argument('--holdout_groups', type=str, nargs='*', default=None,
                        help="Group ids (punch or slide ids, per --split_level) forced into "
                             "the validation subset, e.g. --holdout_groups PVcont to keep "
                             "the annotated punch out of the training pool")
    parser.add_argument('--exclude_groups', type=str, nargs='*', default=None,
                        help="Group ids removed from the dataset entirely (neither train "
                             "nor val), e.g. --exclude_groups RH to drop an out-of-scope "
                             "H&E slide's punch")
    parser.add_argument('--train_slides', type=str, nargs='*', default=None,
                        help="Slide ids forced into the training subset (and removed from "
                             "validation), e.g. --train_slides PVcont9_01 PVcont9_02 to "
                             "teach deep-layer morphology from specific serials")
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval during training')
    parser.add_argument('--visualize_interval', type=int, default=5,
                        help='Interval for visualizing predictions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--shift_type', type=str, default='all',
                        choices=['all', 'no_shift', 'ovlp_20', 'ovlp_40', 'ovlp_60', 'ovlp_80'],
                        help='Type of shift to use for tiled dataset (must match TileGen output)')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs to use for distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='Distributed backend to use')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Use weighted sampling to balance classes')
    parser.add_argument('--sampling_alpha', type=float, default=0.75,
                        help='Alpha value for class weight sampling (0-1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers per GPU (avoid too many to prevent GPU memory pressure)')
    parser.add_argument('--activation_checkpointing', action='store_true', default=False,
                        help='Use activation checkpointing to trade compute for memory (recommended for L4)')
    parser.add_argument('--loss_type', type=str, default='compound',
                        choices=['ce', 'compound', 'focal'],
                        help='Loss function: ce (CrossEntropy), compound (CE+Dice), or focal')
    parser.add_argument('--class_weights', type=str, default='auto',
                        help='Class weight strategy: "none", "auto" (sqrt_inv_freq from data), or comma-separated floats e.g. "0.75,0.71,2.5,4.2"')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['plateau', 'cosine'],
                        help='LR scheduler: plateau (ReduceLROnPlateau) or cosine (CosineAnnealingWarmRestarts)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs for cosine scheduler')
    parser.add_argument('--tile_size', type=int, nargs='+', default=None,
                        help='Disk tile size(s) to load (e.g. 448 768). All are resized to --img_size for the model.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint (.pth) to resume training from')
    parser.add_argument('--use_tta', action='store_true',
                        help='Enable test-time augmentation during validation visualization')
    
    # Weights & Biases arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='lesion-vision',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (team) name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                        help='Weights & Biases tags for this run')
    parser.add_argument('--wandb_log_model', action='store_true',
                        help='Log model as artifact to Weights & Biases')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable cudnn deterministic algorithms and disable benchmark (slower, more reproducible)',
    )

    return parser.parse_args()




def main():
    """Main training function."""
    args = parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Print wandb status
    if args.use_wandb:
        print(f"Weights & Biases tracking enabled. Project: {args.wandb_project}")
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.distributed:
            # Get number of GPUs available
            world_size = min(args.world_size, torch.cuda.device_count())
            print(f"Starting distributed training with {world_size} GPUs")
            
            # Use torchrun-style distributed training
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # Spawn processes
            mp.spawn(
                train_process,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        else:
            # Non-distributed training
            train_process(0, 1, args)
    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
        if args.use_wandb and wandb.run:
            wandb.finish()
        print("Training completed and resources cleaned up.")


if __name__ == "__main__":
    main()
