"""
LSTM Baseline Model for Visit-Grouped EHR Sequences.

This module implements a simple but effective LSTM-based model for learning
from visit-grouped EHR sequences. The model uses:
- Embedding layer for medical codes
- LSTM for temporal modeling across visits
- Optional attention mechanism
- Flexible prediction heads for different tasks
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class VisitEncoder(nn.Module):
    """
    Encode a visit (set of medical codes) into a fixed-size vector.
    
    Supports multiple aggregation strategies:
    - mean: Average of code embeddings
    - sum: Sum of code embeddings
    - max: Max pooling over code embeddings
    - attention: Learned attention weights over codes
    """
    
    def __init__(
        self,
        embedding_dim: int,
        aggregation: str = 'mean',
        dropout: float = 0.1
    ):
        """
        Initialize visit encoder.
        
        Args:
            embedding_dim: Dimension of code embeddings
            aggregation: How to aggregate codes ('mean', 'sum', 'max', 'attention')
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.dropout = nn.Dropout(dropout)
        
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.Tanh(),
                nn.Linear(embedding_dim // 2, 1)
            )
    
    def forward(
        self,
        visit_embeddings: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode visit into single vector.
        
        Args:
            visit_embeddings: [batch_size, max_codes_per_visit, embedding_dim]
            visit_mask: [batch_size, max_codes_per_visit] - 1 for real codes, 0 for padding
        
        Returns:
            visit_vector: [batch_size, embedding_dim]
        """
        if visit_mask is None:
            visit_mask = torch.ones(
                visit_embeddings.shape[:-1],
                device=visit_embeddings.device
            )
        
        # Expand mask for broadcasting
        visit_mask = visit_mask.unsqueeze(-1)  # [batch, codes, 1]
        
        if self.aggregation == 'mean':
            # Masked mean
            masked_embeddings = visit_embeddings * visit_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            count = visit_mask.sum(dim=1).clamp(min=1)
            visit_vector = sum_embeddings / count
            
        elif self.aggregation == 'sum':
            # Masked sum
            masked_embeddings = visit_embeddings * visit_mask
            visit_vector = masked_embeddings.sum(dim=1)
            
        elif self.aggregation == 'max':
            # Masked max pooling
            masked_embeddings = visit_embeddings.masked_fill(
                visit_mask == 0, float('-inf')
            )
            visit_vector = masked_embeddings.max(dim=1)[0]
            
        elif self.aggregation == 'attention':
            # Learned attention weights
            attention_scores = self.attention(visit_embeddings)  # [batch, codes, 1]
            attention_scores = attention_scores.masked_fill(
                visit_mask == 0, float('-inf')
            )
            attention_weights = torch.softmax(attention_scores, dim=1)
            visit_vector = (visit_embeddings * attention_weights).sum(dim=1)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return self.dropout(visit_vector)


class LSTMBaseline(nn.Module):
    """
    LSTM-based model for visit-grouped EHR sequences.
    
    Architecture:
    1. Embedding layer: Maps medical codes to dense vectors
    2. Visit encoder: Aggregates codes within each visit
    3. LSTM: Models temporal dependencies across visits
    4. Prediction head: Task-specific output layer
    
    Supports multiple tasks:
    - Binary classification (e.g., disease prediction)
    - Multi-class classification (e.g., phenotyping)
    - Regression (e.g., risk scores)
    - Sequence prediction (e.g., next visit prediction)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        visit_aggregation: str = 'mean',
        output_dim: int = 1,
        task: str = 'binary_classification',
        padding_idx: int = 0
    ):
        """
        Initialize LSTM baseline model.
        
        Args:
            vocab_size: Size of medical code vocabulary
            embedding_dim: Dimension of code embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            visit_aggregation: How to aggregate codes within visits
            output_dim: Output dimension (1 for binary, K for K-class)
            task: Task type ('binary_classification', 'multi_class', 'regression')
            padding_idx: Index for padding token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.task = task
        self.output_dim = output_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Visit encoder
        self.visit_encoder = VisitEncoder(
            embedding_dim=embedding_dim,
            aggregation=visit_aggregation,
            dropout=dropout
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension of LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Prediction head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # Task-specific activation
        if task == 'binary_classification':
            self.activation = nn.Sigmoid()
        elif task == 'multi_class':
            self.activation = nn.Softmax(dim=-1)
        else:  # regression
            self.activation = nn.Identity()
    
    def forward(
        self,
        visit_codes: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            visit_codes: [batch_size, num_visits, max_codes_per_visit]
                Medical codes for each visit
            visit_mask: [batch_size, num_visits, max_codes_per_visit]
                Mask for padding codes (1 = real, 0 = padding)
            sequence_mask: [batch_size, num_visits]
                Mask for padding visits (1 = real, 0 = padding)
            return_hidden: Whether to return LSTM hidden states
        
        Returns:
            Dictionary containing:
                - logits: [batch_size, output_dim] - Raw predictions
                - predictions: [batch_size, output_dim] - After activation
                - hidden_states: [batch_size, num_visits, hidden_dim] (if return_hidden)
        """
        batch_size, num_visits, max_codes = visit_codes.shape
        
        # Embed all codes
        # [batch, visits, codes] -> [batch, visits, codes, embed_dim]
        code_embeddings = self.embedding(visit_codes)
        
        # Encode each visit
        # Reshape to process all visits at once
        code_embeddings_flat = code_embeddings.view(
            batch_size * num_visits, max_codes, self.embedding_dim
        )
        
        if visit_mask is not None:
            visit_mask_flat = visit_mask.view(batch_size * num_visits, max_codes)
        else:
            visit_mask_flat = None
        
        # Get visit representations
        visit_vectors = self.visit_encoder(code_embeddings_flat, visit_mask_flat)
        visit_vectors = visit_vectors.view(batch_size, num_visits, self.embedding_dim)
        
        # Apply LSTM across visits
        if sequence_mask is not None:
            # Pack padded sequence for efficiency
            lengths = sequence_mask.sum(dim=1).cpu()
            packed_input = nn.utils.rnn.pack_padded_sequence(
                visit_vectors,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_input)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(visit_vectors)
        
        # Use last hidden state for prediction
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        
        # Prediction
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        predictions = self.activation(logits)
        
        output = {
            'logits': logits,
            'predictions': predictions
        }
        
        if return_hidden:
            output['hidden_states'] = lstm_output
        
        return output
    
    def get_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for medical codes.
        
        Args:
            codes: [batch_size, num_codes] - Medical code indices
        
        Returns:
            embeddings: [batch_size, num_codes, embedding_dim]
        """
        return self.embedding(codes)
    
    def predict(
        self,
        visit_codes: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions (convenience method).
        
        Args:
            visit_codes: [batch_size, num_visits, max_codes_per_visit]
            visit_mask: [batch_size, num_visits, max_codes_per_visit]
            sequence_mask: [batch_size, num_visits]
        
        Returns:
            predictions: [batch_size, output_dim]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(visit_codes, visit_mask, sequence_mask)
        return output['predictions']


def create_lstm_baseline(
    vocab_size: int,
    task: str = 'binary_classification',
    output_dim: Optional[int] = None,
    model_size: str = 'small'
) -> LSTMBaseline:
    """
    Factory function to create LSTM baseline with preset configurations.
    
    Args:
        vocab_size: Size of medical code vocabulary
        task: Task type ('binary_classification', 'multi_class', 'regression')
        output_dim: Output dimension (inferred from task if not provided)
        model_size: Model size ('small', 'medium', 'large')
    
    Returns:
        Configured LSTMBaseline model
    """
    # Infer output_dim if not provided
    if output_dim is None:
        if task == 'binary_classification':
            output_dim = 1
        elif task == 'regression':
            output_dim = 1
        else:
            raise ValueError("output_dim must be specified for multi_class task")
    
    # Model size configurations
    configs = {
        'small': {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 1,
            'dropout': 0.3
        },
        'medium': {
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.3
        },
        'large': {
            'embedding_dim': 512,
            'hidden_dim': 1024,
            'num_layers': 3,
            'dropout': 0.4
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    
    return LSTMBaseline(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        output_dim=output_dim,
        task=task
    )
