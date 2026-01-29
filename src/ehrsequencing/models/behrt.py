"""
BEHRT: BERT for Electronic Health Records

Implementation of BEHRT (Li et al., 2019) - a BERT-based model for learning
patient representations from EHR sequences with temporal embeddings.

Key features:
- Age + visit + position embeddings for temporal modeling
- Transformer encoder for sequence processing
- Support for 3 model sizes (small/medium/large)
- Pre-training with MLM and next visit prediction
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from .embeddings import BEHRTEmbedding


@dataclass
class BEHRTConfig:
    """Configuration for BEHRT model."""
    
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    intermediate_dim: Optional[int] = None  # Default: 4 * hidden_dim
    max_position: int = 512
    max_age: int = 100
    max_visits: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_sinusoidal: bool = False
    
    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim
        
        # Validate dimensions
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
    
    @classmethod
    def small(cls, vocab_size: int) -> 'BEHRTConfig':
        """Small model config for local development (M1 16GB)."""
        return cls(
            vocab_size=vocab_size,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            max_position=50,
            dropout=0.1
        )
    
    @classmethod
    def medium(cls, vocab_size: int) -> 'BEHRTConfig':
        """Medium model config for local/small GPU."""
        return cls(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            max_position=100,
            dropout=0.1
        )
    
    @classmethod
    def large(cls, vocab_size: int) -> 'BEHRTConfig':
        """Large model config for cloud GPU (A40)."""
        return cls(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            max_position=200,
            dropout=0.1
        )


class BEHRT(nn.Module):
    """
    BEHRT: BERT for Electronic Health Records.
    
    Transformer-based model for learning patient representations from EHR sequences.
    Uses age, visit, and position embeddings to capture temporal information.
    
    Args:
        config: BEHRTConfig object with model hyperparameters
    
    Example:
        >>> config = BEHRTConfig.small(vocab_size=1000)
        >>> model = BEHRT(config)
        >>> codes = torch.randint(0, 1000, (32, 50))  # [batch, seq_len]
        >>> ages = torch.randint(0, 100, (32, 50))
        >>> visit_ids = torch.arange(50).unsqueeze(0).expand(32, -1)
        >>> mask = torch.ones(32, 50, dtype=torch.bool)
        >>> output = model(codes, ages, visit_ids, mask)
        >>> output.shape  # [32, 50, 128]
    """
    
    def __init__(self, config: BEHRTConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embeddings = BEHRTEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            max_age=config.max_age,
            max_visits=config.max_visits,
            max_position=config.max_position,
            dropout=config.dropout,
            use_sinusoidal=config.use_sinusoidal
        )
        
        # Project embeddings to hidden dimension if different
        if config.embedding_dim != config.hidden_dim:
            self.embedding_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        else:
            self.embedding_projection = nn.Identity()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN like modern transformers
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through BEHRT.
        
        Args:
            codes: Medical code IDs. Shape: [batch_size, seq_length]
            ages: Patient ages at each position. Shape: [batch_size, seq_length]
            visit_ids: Visit IDs (0-indexed). Shape: [batch_size, seq_length]
            attention_mask: Mask for valid positions (1=valid, 0=padding).
                          Shape: [batch_size, seq_length]
        
        Returns:
            Sequence representations. Shape: [batch_size, seq_length, hidden_dim]
        """
        # Get embeddings
        embeddings = self.embeddings(codes, ages, visit_ids)
        
        # Project to hidden dimension
        hidden_states = self.embedding_projection(embeddings)
        
        # Create attention mask for transformer (True = masked position)
        if attention_mask is not None:
            # Convert from (1=valid, 0=padding) to (True=padding, False=valid)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Pass through transformer encoder
        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    
    def get_patient_embedding(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling: str = 'cls'
    ) -> torch.Tensor:
        """
        Get patient-level embedding from sequence.
        
        Args:
            codes: Medical code IDs. Shape: [batch_size, seq_length]
            ages: Patient ages. Shape: [batch_size, seq_length]
            visit_ids: Visit IDs. Shape: [batch_size, seq_length]
            attention_mask: Mask for valid positions. Shape: [batch_size, seq_length]
            pooling: Pooling strategy ('cls', 'mean', 'max')
        
        Returns:
            Patient embeddings. Shape: [batch_size, hidden_dim]
        """
        # Get sequence representations
        hidden_states = self.forward(codes, ages, visit_ids, attention_mask)
        
        if pooling == 'cls':
            # Use first token (CLS) representation
            patient_embedding = hidden_states[:, 0, :]
        elif pooling == 'mean':
            # Mean pooling over valid positions
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1)
                patient_embedding = sum_embeddings / sum_mask
            else:
                patient_embedding = hidden_states.mean(dim=1)
        elif pooling == 'max':
            # Max pooling over valid positions
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states_masked = hidden_states.masked_fill(mask_expanded == 0, float('-inf'))
                patient_embedding = hidden_states_masked.max(dim=1)[0]
            else:
                patient_embedding = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return patient_embedding
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BEHRTForMLM(nn.Module):
    """
    BEHRT with Masked Language Modeling head for pre-training.
    
    Predicts masked medical codes from context using a linear projection
    followed by softmax over the vocabulary.
    
    Args:
        config: BEHRTConfig object
    """
    
    def __init__(self, config: BEHRTConfig):
        super().__init__()
        self.config = config
        self.behrt = BEHRT(config)
        
        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )
        
    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional MLM loss computation.
        
        Args:
            codes: Medical code IDs (may contain [MASK] tokens). Shape: [batch, seq_len]
            ages: Patient ages. Shape: [batch, seq_len]
            visit_ids: Visit IDs. Shape: [batch, seq_len]
            attention_mask: Mask for valid positions. Shape: [batch, seq_len]
            labels: True code IDs for masked positions (-100 for non-masked).
                   Shape: [batch, seq_len]
        
        Returns:
            logits: Predictions for all positions. Shape: [batch, seq_len, vocab_size]
            loss: MLM loss if labels provided, else None
        """
        # Get sequence representations
        hidden_states = self.behrt(codes, ages, visit_ids, attention_mask)
        
        # Predict codes
        logits = self.mlm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return logits, loss


class BEHRTForNextVisitPrediction(nn.Module):
    """
    BEHRT with next visit prediction head for pre-training.
    
    Predicts codes in the next visit given history. Uses multi-label
    classification since visits can contain multiple codes.
    
    Args:
        config: BEHRTConfig object
    """
    
    def __init__(self, config: BEHRTConfig):
        super().__init__()
        self.config = config
        self.behrt = BEHRT(config)
        
        # Next visit prediction head
        self.nvp_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )
        
    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional next visit prediction loss.
        
        Args:
            codes: Medical code IDs. Shape: [batch, seq_len]
            ages: Patient ages. Shape: [batch, seq_len]
            visit_ids: Visit IDs. Shape: [batch, seq_len]
            attention_mask: Mask for valid positions. Shape: [batch, seq_len]
            labels: Multi-hot labels for next visit codes. Shape: [batch, vocab_size]
        
        Returns:
            logits: Predictions for next visit. Shape: [batch, vocab_size]
            loss: Binary cross-entropy loss if labels provided, else None
        """
        # Get patient embedding (use CLS token)
        patient_embedding = self.behrt.get_patient_embedding(
            codes, ages, visit_ids, attention_mask, pooling='cls'
        )
        
        # Predict next visit codes
        logits = self.nvp_head(patient_embedding)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return logits, loss


class BEHRTForSequenceClassification(nn.Module):
    """
    BEHRT with classification head for downstream tasks.
    
    Can be used for tasks like diagnosis prediction, readmission prediction, etc.
    
    Args:
        config: BEHRTConfig object
        num_labels: Number of output classes
        pooling: Pooling strategy for patient embedding ('cls', 'mean', 'max')
    """
    
    def __init__(self, config: BEHRTConfig, num_labels: int, pooling: str = 'cls'):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.pooling = pooling
        
        self.behrt = BEHRT(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_labels)
        )
        
    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional classification loss.
        
        Args:
            codes: Medical code IDs. Shape: [batch, seq_len]
            ages: Patient ages. Shape: [batch, seq_len]
            visit_ids: Visit IDs. Shape: [batch, seq_len]
            attention_mask: Mask for valid positions. Shape: [batch, seq_len]
            labels: Class labels. Shape: [batch] for single-label, [batch, num_labels] for multi-label
        
        Returns:
            logits: Classification logits. Shape: [batch, num_labels]
            loss: Classification loss if labels provided, else None
        """
        # Get patient embedding
        patient_embedding = self.behrt.get_patient_embedding(
            codes, ages, visit_ids, attention_mask, pooling=self.pooling
        )
        
        # Classify
        logits = self.classifier(patient_embedding)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif labels.dim() == 1:
                # Single-label classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:
                # Multi-label classification
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
        
        return logits, loss
