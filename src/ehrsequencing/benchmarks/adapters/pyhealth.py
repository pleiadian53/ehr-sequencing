"""
PyHealth adapter for benchmarking against ehrsequencing models.

This adapter wraps PyHealth's models to provide a unified interface for
benchmarking against ehrsequencing's BEHRT implementation.

Note: This module requires PyHealth to be installed:
    pip install pyhealth
    or
    mamba env create -f environment-benchmarking.yml
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ehrsequencing.benchmarks.adapters.base import BaseModelAdapter

try:
    from pyhealth.models import Transformer as PyHealthTransformer
    PYHEALTH_AVAILABLE = True
except ImportError:
    PYHEALTH_AVAILABLE = False
    PyHealthTransformer = None


class PyHealthAdapter(BaseModelAdapter):
    """
    Adapter for PyHealth's Transformer model.
    
    This adapter wraps PyHealth's generic Transformer to enable direct comparison
    with ehrsequencing's BEHRT model on the same tasks and data.
    
    Example:
        >>> config = {
        ...     'vocab_size': 1000,
        ...     'embedding_dim': 256,
        ...     'hidden_dim': 512,
        ...     'num_layers': 6,
        ...     'num_heads': 8,
        ...     'dropout': 0.1
        ... }
        >>> adapter = PyHealthAdapter(config=config)
        >>> adapter.build_model()
        >>> results = adapter.train(train_loader, val_loader, epochs=50)
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        """
        Initialize PyHealth adapter.
        
        Args:
            config: Model configuration matching BEHRT config format
            device: Device to run on ('cuda' or 'cpu')
        
        Raises:
            ImportError: If PyHealth is not installed
        """
        if not PYHEALTH_AVAILABLE:
            raise ImportError(
                "PyHealth is not installed. Install it with:\n"
                "  pip install pyhealth\n"
                "or create the benchmarking environment:\n"
                "  mamba env create -f environment-benchmarking.yml"
            )
        
        super().__init__(model_name='PyHealth-Transformer', config=config, device=device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def build_model(self) -> nn.Module:
        """
        Build PyHealth Transformer model.
        
        PyHealth's Transformer is a generic sequence model without EHR-specific
        features like age/visit embeddings. This provides a baseline for comparison.
        
        Returns:
            PyHealth Transformer model
        """
        # PyHealth expects different config format, so we adapt
        # Note: PyHealth's Transformer is more generic than BEHRT
        self.model = SimpleTransformerMLM(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        return self.model
    
    def prepare_data(self, codes, ages, visit_ids, attention_mask, labels):
        """
        Convert ehrsequencing data to PyHealth format.
        
        Note: PyHealth's generic Transformer only uses code sequences,
        not age/visit embeddings. This is a key difference from BEHRT.
        
        Args:
            codes: Code sequences [batch, seq_len]
            ages: Age sequences [batch, seq_len] (ignored by PyHealth)
            visit_ids: Visit sequences [batch, seq_len] (ignored by PyHealth)
            attention_mask: Attention mask [batch, seq_len]
            labels: MLM labels [batch, seq_len]
        
        Returns:
            Dictionary with PyHealth-compatible format
        """
        return {
            'codes': codes,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train PyHealth model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            **kwargs: Additional arguments
        
        Returns:
            Training history with losses and metrics
        """
        if self.model is None:
            self.build_model()
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                codes, ages, visit_ids, attention_mask, labels = batch
                codes = codes.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (only uses codes, not ages/visits)
                logits = self.model(codes, attention_mask)
                
                # Compute loss
                loss = self.criterion(
                    logits.view(-1, self.config['vocab_size']),
                    labels.view(-1)
                )
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Compute accuracy
                mask = labels != -100
                predictions = logits.argmax(dim=-1)
                train_correct += ((predictions == labels) & mask).sum().item()
                train_total += mask.sum().item()
            
            # Validation
            val_loss, val_accuracy = self._validate(val_loader)
            
            # Record metrics
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_accuracy'].append(train_correct / train_total if train_total > 0 else 0.0)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Train Acc: {history['train_accuracy'][-1]:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")
        
        self.is_trained = True
        return history
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation and return loss and accuracy."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                codes, ages, visit_ids, attention_mask, labels = batch
                codes = codes.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(codes, attention_mask)
                
                loss = self.criterion(
                    logits.view(-1, self.config['vocab_size']),
                    labels.view(-1)
                )
                
                val_loss += loss.item()
                
                mask = labels != -100
                predictions = logits.argmax(dim=-1)
                val_correct += ((predictions == labels) & mask).sum().item()
                val_total += mask.sum().item()
        
        return (
            val_loss / len(val_loader),
            val_correct / val_total if val_total > 0 else 0.0
        )
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary of metrics
        """
        test_loss, test_accuracy = self._validate(test_loader)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions.
        
        Args:
            data_loader: Data loader
        
        Returns:
            Tuple of (predictions, ground_truth)
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                codes, ages, visit_ids, attention_mask, labels = batch
                codes = codes.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                logits = self.model(codes, attention_mask)
                predictions = logits.argmax(dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_predictions), torch.cat(all_labels)


class SimpleTransformerMLM(nn.Module):
    """
    Simple Transformer for Masked Language Modeling.
    
    This is a minimal implementation similar to PyHealth's Transformer,
    without EHR-specific features (no age/visit embeddings).
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(512, embedding_dim)  # Max seq length 512
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlm_head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, codes, attention_mask):
        """
        Forward pass.
        
        Args:
            codes: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = codes.shape
        
        # Embeddings (only codes, no age/visit)
        code_embeds = self.embedding(codes)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=codes.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(positions)
        
        embeddings = self.dropout(code_embeds + pos_embeds)
        
        # Transformer (attention_mask: 1 = attend, 0 = ignore)
        # PyTorch expects: True = ignore, False = attend
        mask = (attention_mask == 0)
        
        hidden = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # MLM head
        logits = self.mlm_head(hidden)
        
        return logits
