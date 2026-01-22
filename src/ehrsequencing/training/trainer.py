"""
Training utilities for EHR sequence models.

Provides a flexible Trainer class for training and evaluating models
with support for:
- Multiple loss functions and metrics
- Early stopping and learning rate scheduling
- Checkpointing and model saving
- Logging and progress tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer for EHR sequence models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        metrics: Optional[Dict[str, Callable]] = None,
        scheduler: Optional[Any] = None,
        early_stopping_patience: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on ('cpu', 'cuda', 'mps')
            metrics: Dictionary of metric functions {name: function}
            scheduler: Learning rate scheduler
            early_stopping_patience: Patience for early stopping (None = disabled)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics or {}
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            **{f'train_{name}': [] for name in self.metrics.keys()},
            **{f'val_{name}': [] for name in self.metrics.keys()}
        }
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics.keys()}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                batch['visit_codes'],
                visit_mask=batch.get('visit_mask'),
                sequence_mask=batch.get('sequence_mask')
            )
            
            # Compute loss
            loss = self.criterion(outputs['logits'], batch['labels'])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Compute additional metrics
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(
                        outputs['predictions'].cpu(),
                        batch['labels'].cpu()
                    )
                    metric_values[name] += metric_value
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches 
                      for name, value in metric_values.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        metric_values = {name: 0 for name in self.metrics.keys()}
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch} [Val]')
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['visit_codes'],
                    visit_mask=batch.get('visit_mask'),
                    sequence_mask=batch.get('sequence_mask')
                )
                
                # Compute loss
                loss = self.criterion(outputs['logits'], batch['labels'])
                total_loss += loss.item()
                num_batches += 1
                
                # Compute metrics
                for name, metric_fn in self.metrics.items():
                    metric_value = metric_fn(
                        outputs['predictions'].cpu(),
                        batch['labels'].cpu()
                    )
                    metric_values[name] += metric_value
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches 
                      for name, value in metric_values.items()}
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            for name, value in train_metrics.items():
                if name != 'loss':
                    self.history[f'train_{name}'].append(value)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                for name, value in val_metrics.items():
                    if name != 'loss':
                        self.history[f'val_{name}'].append(value)
                
                # Log metrics
                logger.info(
                    f"Epoch {self.current_epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if self.checkpoint_dir:
                        self.save_checkpoint('best_model.pt')
                else:
                    self.epochs_without_improvement += 1
                    
                    if (self.early_stopping_patience is not None and 
                        self.epochs_without_improvement >= self.early_stopping_patience):
                        logger.info(
                            f"Early stopping triggered after {self.current_epoch} epochs"
                        )
                        break
            else:
                logger.info(
                    f"Epoch {self.current_epoch}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}"
                )
            
            # Save checkpoint every epoch
            if self.checkpoint_dir:
                self.save_checkpoint(f'checkpoint_epoch_{self.current_epoch}.pt')
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set")
        
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


def binary_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute binary classification accuracy.
    
    Args:
        predictions: [batch_size, 1] - Predicted probabilities
        labels: [batch_size, 1] - True labels (0 or 1)
    
    Returns:
        Accuracy as float
    """
    pred_labels = (predictions > 0.5).float()
    correct = (pred_labels == labels).float().sum()
    return (correct / labels.numel()).item()


def auroc(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute AUROC (Area Under ROC Curve).
    
    Args:
        predictions: [batch_size, 1] - Predicted probabilities
        labels: [batch_size, 1] - True labels (0 or 1)
    
    Returns:
        AUROC as float
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(
            labels.numpy().flatten(),
            predictions.numpy().flatten()
        )
    except ImportError:
        logger.warning("sklearn not available, skipping AUROC computation")
        return 0.0
    except ValueError:
        # Not enough classes in labels
        return 0.0
