"""Data loaders, train/validate orchestration, per-rank process."""

import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb

from src.data.dataset import get_train_val_dataset
from src.models.segmentation_model import LesionSegmentationModel
from src.train.distributed import cleanup, setup
from src.train.viz import create_color_mask, plot_training_metrics, visualize_results


def create_data_loaders(args, rank=0, world_size=1):
    """Create data loaders for training and validation sets using tiled dataset."""
    # Set random seed for reproducibility
    seed = args.seed + rank  # Different seed per process
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    tile_sizes = args.tile_size if args.tile_size else [args.img_size]
    if rank == 0:
        label = "+".join(str(s) for s in tile_sizes)
        print(f"Loading tile sizes {label} from disk -> resize to {args.img_size}px for model")

    train_dataset, val_dataset = get_train_val_dataset(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        shift_type=args.shift_type,
        augment=args.augment,
        val_split=args.val_split,
        random_state=args.seed,
        disk_tile_sizes=tile_sizes,
    )

    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    # Generate samplers and data loaders
    train_sampler = None
    
    if args.distributed:
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=False,
            seed=args.seed
        )
    elif args.weighted_sampling:
        # Use weighted sampling for balanced classes
        if rank == 0:
            print("Computing sample weights for balanced training...")
        train_sampler = train_dataset.get_weighted_sampler(
            alpha=args.sampling_alpha,
            num_analyze=1000
        )
    
    num_workers = args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=args.distributed,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler if args.distributed else None,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return train_loader, val_loader, train_sampler


def train_model(model, train_loader, val_loader, train_sampler, args, rank=0):
    """Train the segmentation model."""
    # Make sure output directory exists
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Define model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"model_{timestamp}")
    
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
    
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    # Save training configuration
    if rank == 0:
        config = {
            'backbone': args.backbone,
            'img_size': args.img_size,
            'tile_sizes': args.tile_size or [args.img_size],
            'batch_size': args.batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'amp': args.amp,
            'activation_checkpointing': args.activation_checkpointing,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'n_classes': args.n_classes,
            'augment': args.augment,
            'shift_type': args.shift_type,
            'timestamp': timestamp,
            'distributed': args.distributed,
            'world_size': args.world_size,
            'weighted_sampling': args.weighted_sampling,
            'sampling_alpha': args.sampling_alpha,
            'num_workers': args.num_workers,
            'loss_type': args.loss_type,
            'class_weights': args.class_weights,
            'scheduler_type': args.scheduler_type,
            'warmup_epochs': args.warmup_epochs,
            'resumed_from': args.resume,
            'deterministic': args.deterministic,
            'seed': args.seed,
        }
        
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb_run_name = args.wandb_name if args.wandb_name else f"model_{timestamp}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                tags=args.wandb_tags,
                config=config
            )
    
    # Define a custom callback function for wandb logging
    def log_metrics_callback(epoch, metrics, rank):
        if rank == 0 and args.use_wandb:
            # Log training metrics to wandb
            wandb.log(metrics, step=epoch)
            
            # Every few epochs, log validation prediction examples if available
            if epoch % args.visualize_interval == 0 and 'val_predictions' in metrics:
                val_imgs = metrics['val_predictions'].get('images', [])
                val_gt = metrics['val_predictions'].get('ground_truth', [])
                val_pred = metrics['val_predictions'].get('predictions', [])
                
                if len(val_imgs) > 0 and len(val_gt) > 0 and len(val_pred) > 0:
                    for i in range(min(3, len(val_imgs))):
                        # Transpose image from CHW [3, H, W] to HWC [H, W, 3] format for wandb
                        img_hwc = val_imgs[i].transpose(1, 2, 0)
                        
                        wandb.log({
                            f"val_example_{i}": wandb.Image(
                                img_hwc,  # Use transposed image
                                masks={
                                    "ground_truth": {"mask_data": val_gt[i], "class_labels": model.class_names},
                                    "prediction": {"mask_data": val_pred[i], "class_labels": model.class_names}
                                }
                            )
                        }, step=epoch)
    
    # Train the model (AMP and gradient accumulation handled inside model.train)
    history = model.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_sampler=train_sampler,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_save_path=best_model_path if rank == 0 else None,
        patience=args.patience,
        log_interval=args.log_interval,
        rank=rank,
        distributed=args.distributed,
        callback=log_metrics_callback if args.use_wandb else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.amp,
    )
    
    # Save the trained model
    if rank == 0:
        model.save(best_model_path)
        print(f"Model saved to {best_model_path}")
        
        # Plot training history
        plot_training_metrics(history, save_path=os.path.join(model_dir, 'training_metrics.png'))
        
        # Log model as artifact to wandb
        if args.use_wandb and args.wandb_log_model:
            artifact = wandb.Artifact(f"model_{timestamp}", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
    
    return model, history


def validate_model(model, val_loader, args, rank=0):
    """Validate the model on the validation set and visualize results."""
    # Skip if not main process
    if rank != 0:
        return
        
    # Create a table for wandb logging
    if args.use_wandb:
        validation_table = wandb.Table(columns=["Sample", "Image", "Ground Truth", "Prediction"])
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization directory
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate on some validation samples
    if args.distributed:
        model.module.eval()
    else:
        model.model.eval()
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            # Only visualize a few samples
            if i >= 5:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Make predictions
            if args.distributed:
                outputs = model(images)
            else:
                outputs = model.model(images)
            
            # Convert to probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predicted classes
            _, predictions = torch.max(probabilities, dim=1)
            
            # Convert to numpy for visualization
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            
            # Visualize each image in the batch
            for b in range(images.shape[0]):
                # Get the original image, ground truth, and prediction
                image = images_np[b].transpose(1, 2, 0)  # Convert from [C,H,W] to [H,W,C]
                image = np.clip(image, 0, 1)  # Ensure values in [0, 1]
                
                # Create color masks
                gt_mask = create_color_mask(masks_np[b])
                pred_mask = create_color_mask(predictions_np[b])
                
                # Visualize results
                fig = visualize_results(
                    image=image, 
                    gt_mask=gt_mask, 
                    pred_mask=pred_mask,
                    title=f"Validation Sample {i*args.batch_size+b}"
                )
                
                # Save visualization
                fig.savefig(os.path.join(vis_dir, f"val_sample_{i*args.batch_size+b}.png"))
                
                # Log to wandb
                if args.use_wandb:
                    # Add to validation table
                    sample_id = f"val_sample_{i*args.batch_size+b}"
                    validation_table.add_data(
                        sample_id,
                        wandb.Image(image),
                        wandb.Image(gt_mask),
                        wandb.Image(pred_mask)
                    )
                    
                plt.close(fig)




def train_process(rank, world_size, args):
    """
    Training process for distributed training.
    """
    try:
        # Set up distributed environment only if distributed training is enabled
        if args.distributed and world_size > 1:
            setup(rank, world_size, args)
        
        # Only initialize wandb on the main process
        if rank == 0 and args.use_wandb and not wandb.run:
            print("Initializing Weights & Biases logging")
        
        # Set device
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Empty GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create data loaders
        train_loader, val_loader, train_sampler = create_data_loaders(args, rank, world_size)
        
        # Resolve class weights
        resolved_class_weights = None
        if args.class_weights == 'auto':
            if rank == 0:
                print("Computing class weights from training data...")
                resolved_class_weights = train_loader.dataset.compute_class_weights(
                    num_samples=2000, method='sqrt_inv_freq'
                )
            if args.distributed:
                import torch.distributed as dist
                if resolved_class_weights is not None:
                    cw_tensor = torch.tensor(resolved_class_weights, dtype=torch.float32).cuda(rank)
                else:
                    cw_tensor = torch.zeros(args.n_classes, dtype=torch.float32).cuda(rank)
                dist.broadcast(cw_tensor, src=0)
                resolved_class_weights = cw_tensor.cpu().numpy()
        elif args.class_weights != 'none':
            resolved_class_weights = np.array([float(x) for x in args.class_weights.split(',')])
            if rank == 0:
                print(f"Using manual class weights: {resolved_class_weights}")
        
        # Create model (exactly one process per GPU; batch moved to device in training loop)
        model = LesionSegmentationModel(
            n_classes=args.n_classes,
            backbone=args.backbone,
            encoder_weights='imagenet',
            learning_rate=args.learning_rate,
            input_shape=(3, args.img_size, args.img_size),
            device=device,
            use_amp=args.amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            activation_checkpointing=args.activation_checkpointing,
            loss_type=args.loss_type,
            class_weights=resolved_class_weights,
            scheduler_type=args.scheduler_type,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
        )

        if args.resume:
            if rank == 0:
                print(f"Resuming from checkpoint: {args.resume}")
            model.load(args.resume, device=device)
            if rank == 0:
                print("Checkpoint loaded — model weights, optimizer and scheduler restored")
        
        # Set up distributed training
        if args.distributed:
            try:
                model.setup_for_distributed(rank, world_size)
            except Exception as e:
                print(f"[Rank {rank}] Failed to setup distributed model: {e}")
                raise
        
        # Train model
        model, history = train_model(model, train_loader, val_loader, train_sampler, args, rank=rank)
        
        # Validate model (only on rank 0)
        if rank == 0:
            validate_model(model, val_loader, args, rank=rank)
            
            # Finish wandb run if enabled
            if args.use_wandb and wandb.run:
                wandb.finish()
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up
        cleanup()


