"""Epoch train / validate loop mixed into LesionSegmentationModel."""

import os
import time

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm


class TrainEngine:
    def train_one_epoch(self, dataloader, rank=0, distributed=False, gradient_accumulation_steps=None, use_amp=None):
        """
        Train the model for one epoch.
        Uses mixed precision (autocast + GradScaler) and gradient accumulation when enabled.
        """
        model = self.model
        model.train()
        accum_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else self.gradient_accumulation_steps
        use_amp = use_amp if use_amp is not None else self.use_amp
        scaler = self.scaler if use_amp else None
        
        total_loss = 0.0
        correct = 0
        total_pixels = 0
        batch_count = 0
        
        if rank == 0:
            pbar = tqdm(dataloader, desc="Training", leave=False)
            running_loss = []
            running_acc = []
        else:
            pbar = dataloader
        
        self.optimizer.zero_grad(set_to_none=True)
        for i, (images, masks) in enumerate(pbar):
            # Move inputs to device (Dataset returns CPU tensors; move here only)
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / accum_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Step only after accumulating enough gradients (effective batch = batch_size * accum_steps)
            if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * accum_steps
            batch_count += 1
            
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                batch_correct = (predicted == masks).sum().item()
                batch_total = masks.numel()
                correct += batch_correct
                total_pixels += batch_total
                del predicted
            
            if rank == 0:
                running_loss.append(loss.item() * accum_steps)
                running_acc.append(batch_correct / batch_total)
                if len(running_loss) > 20:
                    running_loss.pop(0)
                    running_acc.pop(0)
                pbar.set_postfix({
                    'loss': f"{np.mean(running_loss):.4f}",
                    'acc': f"{np.mean(running_acc):.4f}"
                })
        
        avg_loss = total_loss / batch_count
        accuracy = correct / total_pixels if total_pixels else 0.0
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, dataloader, rank=0, distributed=False, use_amp=None):
        """Validate the model. Uses autocast when AMP is enabled."""
        model = self.model
        model.eval()
        use_amp = use_amp if use_amp is not None else self.use_amp
        
        total_loss = 0.0
        correct = 0
        total_pixels = 0
        batch_count = 0
        
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validating", leave=False)
        else:
            pbar = dataloader
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = self.criterion(outputs, masks)
                batch_count += 1
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                batch_correct = (predicted == masks).sum().item()
                batch_total = masks.numel()
                correct += batch_correct
                total_pixels += batch_total
                del outputs, predicted
                if rank == 0:
                    pbar.set_postfix({
                        'val_loss': f"{loss.item():.4f}",
                        'val_acc': f"{batch_correct / batch_total:.4f}"
                    })

        avg_loss = total_loss / batch_count
        accuracy = correct / total_pixels if total_pixels else 0.0

        # Allreduce val metrics across ranks so that EVERY rank computes the
        # same val_loss. Without this, each rank early-stops on its own shard's
        # loss, the ranks desynchronize on the next epoch's DDP collectives,
        # and the NCCL watchdog kills the run (SIGABRT after 10-min timeout).
        if distributed:
            import torch.distributed as dist
            t = torch.tensor([total_loss, batch_count, correct, total_pixels],
                             dtype=torch.float64, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, batch_count, correct, total_pixels = t.tolist()
            avg_loss = total_loss / batch_count if batch_count else 0.0
            accuracy = correct / total_pixels if total_pixels else 0.0

        return {'loss': avg_loss, 'accuracy': accuracy}

    def train(
        self,
        train_dataloader,
        val_dataloader,
        train_sampler=None,
        epochs=100,
        learning_rate=1e-4,
        model_save_path=None,
        patience=15,
        log_interval=10,
        rank=0,
        distributed=False,
        callback=None,
        gradient_accumulation_steps=None,
        use_amp=None,
    ):
        """
        Train the model.
        
        Args:
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            train_sampler: Training data sampler for distributed training
            epochs: Number of epochs
            learning_rate: Learning rate
            model_save_path: Path to save best model
            patience: Early stopping patience
            log_interval: Logging interval
            rank: Rank in distributed training
            distributed: Whether using distributed training
        
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        early_stop = False

        # Resume support: continue from the checkpoint's epoch and best val
        # loss so early stopping and the LR schedule continue correctly.
        start_epoch = 0
        resume_state = getattr(self, '_resume_epoch', None)
        if resume_state is not None:
            start_epoch = int(resume_state) + 1
            if getattr(self, '_resume_val_loss', None) is not None:
                best_val_loss = float(self._resume_val_loss)
            if rank == 0:
                print(f"Resuming from epoch {start_epoch} "
                      f"(best val loss {best_val_loss if best_val_loss != float('inf') else 'n/a'})")

        # train_history.json of the resumed run overrides the checkpoint's
        # val_loss (that is the checkpoint epoch's loss, not the best one) and
        # restores the early-stopping counter. Every rank loads it, so the
        # stopping decision stays identical across ranks.
        resume_history = getattr(self, '_resume_history', None)
        epoch_log = []
        best_epoch = None
        if resume_history:
            best_val_loss = float(resume_history['best_val_loss'])
            epochs_no_improve = int(resume_history['epochs_no_improve'])
            best_epoch = int(resume_history['best_epoch'])
            epoch_log = list(resume_history.get('epochs', []))
        history_path = (os.path.join(os.path.dirname(model_save_path), 'train_history.json')
                        if model_save_path else None)
        best_model_ref = (resume_history or {}).get('best_model_path')
        if resume_history and model_save_path and os.path.exists(model_save_path):
            best_model_ref = model_save_path  # previous best, copied into this run dir

        # Create progress bar for epochs on rank 0
        if rank == 0:
            epoch_pbar = tqdm(range(start_epoch, epochs), desc="Epoch Progress",
                              initial=start_epoch, total=epochs, position=0)
        else:
            epoch_pbar = range(start_epoch, epochs)
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # Set epoch for distributed sampler
            if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)
            
            # Train one epoch (AMP and gradient accumulation applied inside)
            train_metrics = self.train_one_epoch(
                train_dataloader, rank, distributed,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_amp=use_amp,
            )
            
            # Validate (AMP for consistency)
            val_metrics = self.validate(val_dataloader, rank, distributed, use_amp=use_amp)
            
            # Update learning rate scheduler
            if self.scheduler_type == 'cosine':
                if epoch >= self.warmup_epochs:
                    self.scheduler.step(epoch - self.warmup_epochs)
            else:
                self.scheduler.step(val_metrics['loss'])
            
            # Save metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            epoch_time = time.time() - epoch_start_time
            
            # Update epoch progress bar on rank 0
            if rank == 0:
                curr_lr = self.optimizer.param_groups[0]['lr']
                epoch_pbar.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}",
                    'lr': f"{curr_lr:.6f}",
                    'time': f"{epoch_time:.1f}s"
                })
                
                # Print detailed metrics at log intervals
                if epoch % log_interval == 0:
                    print(f"\nEpoch {epoch+1}/{epochs} [Time: {epoch_time:.2f}s]")
                    print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                    print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                    print(f"Learning Rate: {curr_lr:.6f}")
            
            # Decide improvement on EVERY rank (val loss is all-reduced, so all
            # ranks agree) and save on rank 0 only. Gating the decision on
            # rank 0 made ranks 1..N-1 count every epoch as "no improvement",
            # so they early-stopped at epoch == patience while rank 0 kept
            # training, and the NCCL watchdog killed the run.
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch + 1
                best_model_ref = model_save_path

                if rank == 0 and model_save_path:
                    if distributed:
                        best_model_state = self.model.module.state_dict().copy()
                    else:
                        best_model_state = self.model.state_dict().copy()
                        
                    torch.save({
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'epoch': epoch,
                        'n_classes': self.n_classes,
                        'backbone': self.backbone,
                        'arch': getattr(self, 'arch', 'auto'),
                        'input_shape': self.input_shape,
                        'learning_rate': self.learning_rate,
                        'loss_type': self.loss_type,
                        'scheduler_type': self.scheduler_type,
                        'warmup_epochs': self.warmup_epochs,
                        'total_epochs': self.total_epochs,
                        'activation_checkpointing': self.activation_checkpointing,
                        'val_loss': best_val_loss,
                        'val_accuracy': val_metrics['accuracy']
                    }, model_save_path)
                    
                    if epoch % log_interval == 0:
                        print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
                        
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Save checkpoint for every epoch on main process
            if rank == 0 and model_save_path:
                # Extract the directory and base filename
                checkpoint_dir = os.path.dirname(model_save_path)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                
                model_state = self.model.module.state_dict().copy() if distributed else self.model.state_dict().copy()
                
                torch.save({
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'n_classes': self.n_classes,
                    'backbone': self.backbone,
                        'arch': getattr(self, 'arch', 'auto'),
                    'input_shape': self.input_shape,
                    'learning_rate': self.learning_rate,
                    'loss_type': self.loss_type,
                    'scheduler_type': self.scheduler_type,
                    'warmup_epochs': self.warmup_epochs,
                    'total_epochs': self.total_epochs,
                    'activation_checkpointing': self.activation_checkpointing,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, checkpoint_path)
                
                if epoch % log_interval == 0:
                    print(f"Saved checkpoint for epoch {epoch+1}")

            # Per-epoch history, usable as --resume_history for a later resume.
            if rank == 0 and history_path:
                import json
                epoch_log.append({'epoch': epoch + 1,
                                  'train_loss': float(train_metrics['loss']),
                                  'val_loss': float(val_metrics['loss']),
                                  'val_accuracy': float(val_metrics['accuracy']),
                                  'checkpoint': os.path.abspath(checkpoint_path)})
                with open(history_path, 'w') as f:
                    json.dump({'patience': patience, 'epochs': epoch_log,
                               'best_epoch': best_epoch, 'best_val_loss': float(best_val_loss),
                               'epochs_no_improve': epochs_no_improve,
                               'best_model_path': os.path.abspath(best_model_ref) if best_model_ref else None,
                               'last_checkpoint': os.path.abspath(checkpoint_path),
                               'resumed_from': (resume_history or {}).get('source_run')}, f, indent=1)

            # Call the callback function with metrics if provided
            if callback is not None:
                # Prepare metrics dictionary for logging
                metrics_to_log = {
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Add visualization examples occasionally
                if epoch % log_interval == 0:
                    # Get a batch of validation data for visualization
                    val_images, val_masks = next(iter(val_dataloader))
                    val_images = val_images.to(self.device)
                    
                    # Make predictions
                    with torch.no_grad():
                        val_outputs = self.model(val_images)
                        val_predictions = torch.softmax(val_outputs, dim=1)
                        _, val_pred_masks = torch.max(val_predictions, dim=1)
                    
                    # Add prediction samples to metrics
                    metrics_to_log['val_predictions'] = {
                        'images': val_images.cpu().numpy(),
                        'ground_truth': val_masks.cpu().numpy(),
                        'predictions': val_pred_masks.cpu().numpy()
                    }
                
                # Call the callback with metrics
                callback(epoch, metrics_to_log, rank)
                
            # Early stopping
            if patience > 0 and epochs_no_improve >= patience:
                if rank == 0:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True
                break
        
        # Restore best model on main process only
        if rank == 0 and best_model_state is not None and model_save_path:
            model = self.model.module if distributed else self.model
            model.load_state_dict(best_model_state)
            
            if epoch % log_interval == 0:
                print(f"Restored best model with Val Loss: {best_val_loss:.4f}")
        
        return history
    
