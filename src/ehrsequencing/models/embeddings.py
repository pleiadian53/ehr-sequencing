"""
Temporal embeddings for EHR sequence models.

This module implements various embedding strategies for incorporating temporal
information into patient sequence models, particularly for BEHRT-style architectures.
"""

import torch
import torch.nn as nn
import math


class AgeEmbedding(nn.Module):
    """
    Age embedding layer that converts continuous age values to discrete embeddings.
    
    Ages are binned into discrete intervals (e.g., 0-5, 5-10, ..., 85+) and then
    embedded. This allows the model to learn age-specific patterns.
    
    Args:
        embedding_dim: Dimension of age embeddings
        max_age: Maximum age to consider (default: 100)
        age_bin_size: Size of age bins in years (default: 5)
    """
    
    def __init__(self, embedding_dim: int, max_age: int = 100, age_bin_size: int = 5):
        super().__init__()
        self.max_age = max_age
        self.age_bin_size = age_bin_size
        self.num_bins = (max_age // age_bin_size) + 2  # +1 for 0, +1 for max+
        
        self.embedding = nn.Embedding(self.num_bins, embedding_dim)
        
    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous ages to embeddings.
        
        Args:
            ages: Tensor of ages in years. Shape: [batch_size, seq_length]
        
        Returns:
            Age embeddings. Shape: [batch_size, seq_length, embedding_dim]
        """
        # Clamp ages to valid range
        ages = torch.clamp(ages, 0, self.max_age)
        
        # Convert to bins
        age_bins = (ages / self.age_bin_size).long()
        
        # Embed
        return self.embedding(age_bins)


class VisitEmbedding(nn.Module):
    """
    Visit embedding layer for sequential visit IDs.
    
    Each visit in a patient's sequence gets a unique embedding based on its
    position in the sequence (1st visit, 2nd visit, etc.).
    
    Args:
        embedding_dim: Dimension of visit embeddings
        max_visits: Maximum number of visits to support (default: 512)
    """
    
    def __init__(self, embedding_dim: int, max_visits: int = 512):
        super().__init__()
        self.max_visits = max_visits
        self.embedding = nn.Embedding(max_visits, embedding_dim)
        
    def forward(self, visit_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert visit IDs to embeddings.
        
        Args:
            visit_ids: Tensor of visit IDs (0-indexed). Shape: [batch_size, seq_length]
        
        Returns:
            Visit embeddings. Shape: [batch_size, seq_length, embedding_dim]
        """
        # Clamp to valid range
        visit_ids = torch.clamp(visit_ids, 0, self.max_visits - 1)
        
        return self.embedding(visit_ids)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings (similar to BERT).
    
    Unlike sinusoidal positional encodings, these are learned during training.
    Each position in the sequence gets a unique embedding.
    
    Args:
        embedding_dim: Dimension of positional embeddings
        max_position: Maximum sequence length (default: 512)
    """
    
    def __init__(self, embedding_dim: int, max_position: int = 512):
        super().__init__()
        self.max_position = max_position
        self.embedding = nn.Embedding(max_position, embedding_dim)
        
    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate positional embeddings for a sequence.
        
        Args:
            seq_length: Length of the sequence
            device: Device to create embeddings on
        
        Returns:
            Positional embeddings. Shape: [1, seq_length, embedding_dim]
        """
        positions = torch.arange(seq_length, device=device).unsqueeze(0)
        return self.embedding(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (from "Attention is All You Need").
    
    Uses sine and cosine functions of different frequencies to encode positions.
    Unlike learnable embeddings, these are fixed and don't require training.
    
    Args:
        embedding_dim: Dimension of positional encodings
        max_position: Maximum sequence length (default: 512)
    """
    
    def __init__(self, embedding_dim: int, max_position: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_position = max_position
        
        # Create positional encoding matrix
        pe = torch.zeros(max_position, embedding_dim)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                            (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_position, embedding_dim]
        
    def forward(self, seq_length: int) -> torch.Tensor:
        """
        Get positional encodings for a sequence.
        
        Args:
            seq_length: Length of the sequence
        
        Returns:
            Positional encodings. Shape: [1, seq_length, embedding_dim]
        """
        return self.pe[:, :seq_length, :]


class TimeEmbedding(nn.Module):
    """
    Time delta embedding for modeling time between events.
    
    Converts continuous time deltas (e.g., days between visits) into embeddings
    using binning strategy similar to age embeddings.
    
    Args:
        embedding_dim: Dimension of time embeddings
        max_time_delta: Maximum time delta to consider in days (default: 365)
        time_bin_size: Size of time bins in days (default: 7)
    """
    
    def __init__(self, embedding_dim: int, max_time_delta: int = 365, time_bin_size: int = 7):
        super().__init__()
        self.max_time_delta = max_time_delta
        self.time_bin_size = time_bin_size
        self.num_bins = (max_time_delta // time_bin_size) + 2
        
        self.embedding = nn.Embedding(self.num_bins, embedding_dim)
        
    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """
        Convert time deltas to embeddings.
        
        Args:
            time_deltas: Tensor of time deltas in days. Shape: [batch_size, seq_length]
        
        Returns:
            Time embeddings. Shape: [batch_size, seq_length, embedding_dim]
        """
        # Clamp to valid range
        time_deltas = torch.clamp(time_deltas, 0, self.max_time_delta)
        
        # Convert to bins
        time_bins = (time_deltas / self.time_bin_size).long()
        
        # Embed
        return self.embedding(time_bins)


class BEHRTEmbedding(nn.Module):
    """
    Combined embedding layer for BEHRT (BERT for EHR).
    
    Combines code embeddings with age, visit, and position embeddings to create
    rich temporal representations for EHR sequences.
    
    Architecture:
        embedding = code_emb + age_emb + visit_emb + position_emb
    
    Args:
        vocab_size: Size of medical code vocabulary
        embedding_dim: Dimension of all embeddings
        max_age: Maximum age in years (default: 100)
        max_visits: Maximum number of visits (default: 512)
        max_position: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
        use_sinusoidal: Use sinusoidal instead of learnable positional embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_age: int = 100,
        max_visits: int = 512,
        max_position: int = 512,
        dropout: float = 0.1,
        use_sinusoidal: bool = False
    ):
        super().__init__()
        
        # Code embeddings
        self.code_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Temporal embeddings
        self.age_embedding = AgeEmbedding(embedding_dim, max_age=max_age)
        self.visit_embedding = VisitEmbedding(embedding_dim, max_visits=max_visits)
        
        if use_sinusoidal:
            self.position_embedding = SinusoidalPositionalEncoding(embedding_dim, max_position)
        else:
            self.position_embedding = PositionalEmbedding(embedding_dim, max_position)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.use_sinusoidal = use_sinusoidal
        
    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Create BEHRT embeddings from codes and temporal information.
        
        Args:
            codes: Medical code IDs. Shape: [batch_size, seq_length]
            ages: Patient ages at each position. Shape: [batch_size, seq_length]
            visit_ids: Visit IDs (0-indexed). Shape: [batch_size, seq_length]
        
        Returns:
            Combined embeddings. Shape: [batch_size, seq_length, embedding_dim]
        """
        batch_size, seq_length = codes.shape
        device = codes.device
        
        # Get individual embeddings
        code_emb = self.code_embedding(codes)
        age_emb = self.age_embedding(ages)
        visit_emb = self.visit_embedding(visit_ids)
        
        if self.use_sinusoidal:
            pos_emb = self.position_embedding(seq_length)
        else:
            pos_emb = self.position_embedding(seq_length, device)
        
        # Combine embeddings (broadcast position embeddings)
        embeddings = code_emb + age_emb + visit_emb + pos_emb
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
