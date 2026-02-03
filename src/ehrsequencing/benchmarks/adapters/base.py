"""
Base adapter interface for external EHR model libraries.

This module defines the abstract interface that all external library adapters
must implement to enable unified benchmarking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader


class BaseModelAdapter(ABC):
    """
    Abstract base class for adapting external EHR model libraries.
    
    This adapter provides a unified interface for training, evaluating, and
    comparing models from different frameworks (PyHealth, TorchEHR, etc.)
    against ehrsequencing models.
    
    Attributes:
        model_name: Name of the model/library being adapted
        config: Configuration dictionary for the model
        device: Device to run the model on (cuda/cpu)
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize the adapter.
        
        Args:
            model_name: Name of the model/library
            config: Configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.config = config
        self.device = device
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and initialize the model.
        
        Returns:
            The initialized model object
        """
        pass
    
    @abstractmethod
    def prepare_data(self, codes, ages, visit_ids, attention_mask, labels) -> Any:
        """
        Convert ehrsequencing data format to library-specific format.
        
        Args:
            codes: Code sequences [batch, seq_len]
            ages: Age sequences [batch, seq_len]
            visit_ids: Visit ID sequences [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Labels for MLM task [batch, seq_len]
        
        Returns:
            Library-specific data format
        """
        pass
    
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            **kwargs: Additional training arguments
        
        Returns:
            Dictionary containing training history and metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions on data.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Tuple of (predictions, ground_truth)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_name': self.model_name,
            'config': self.config,
            'device': self.device,
            'is_trained': self.is_trained,
            'num_parameters': self.count_parameters() if self.model else 0
        }
    
    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        if self.model is None:
            return 0
        
        if isinstance(self.model, torch.nn.Module):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return 0
    
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError(f"save_model not implemented for {self.model_name}")
    
    def load_model(self, path: str):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError(f"load_model not implemented for {self.model_name}")
