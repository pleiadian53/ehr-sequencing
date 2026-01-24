"""
Pretrained medical code embeddings.

This module provides interfaces for loading and using pretrained embeddings
from various sources (Med2Vec, CUI2Vec, clinical BERT, etc.).
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class PretrainedEmbedding(nn.Module):
    """
    Base class for pretrained medical code embeddings.
    
    Handles:
    - Loading pretrained embeddings from disk
    - Mapping vocabulary codes to embedding indices
    - Freezing/unfreezing embeddings during training
    - Handling unknown codes with learned embeddings
    
    Example:
        >>> embedding = PretrainedEmbedding.from_file(
        ...     embedding_path='med2vec_embeddings.pkl',
        ...     vocab=vocab,
        ...     embedding_dim=128
        ... )
        >>> embedded = embedding(code_indices)  # [batch, seq_len, embed_dim]
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int,
        pretrained_weights: Optional[np.ndarray] = None,
        freeze: bool = True,
        padding_idx: int = 0,
    ):
        """
        Initialize pretrained embedding layer.
        
        Args:
            vocab: Code vocabulary (code -> index mapping)
            embedding_dim: Embedding dimension
            pretrained_weights: Pretrained embedding matrix [vocab_size, embed_dim]
            freeze: Whether to freeze pretrained weights during training
            padding_idx: Index for padding token (will be zero vector)
        """
        super().__init__()
        
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.vocab_size = len(vocab)
        self.padding_idx = padding_idx
        self.freeze = freeze
        
        # Create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            self._load_pretrained_weights(pretrained_weights)
        
        # Freeze if requested
        if freeze:
            self.embedding.weight.requires_grad = False
            logger.info("Froze pretrained embedding weights")
    
    def _load_pretrained_weights(self, pretrained_weights: np.ndarray):
        """
        Load pretrained weights into embedding layer.
        
        Args:
            pretrained_weights: Pretrained embedding matrix [vocab_size, embed_dim]
        """
        if pretrained_weights.shape[0] != self.vocab_size:
            logger.warning(
                f"Pretrained weights size {pretrained_weights.shape[0]} "
                f"!= vocab size {self.vocab_size}. Using subset."
            )
            # Truncate or pad as needed
            if pretrained_weights.shape[0] > self.vocab_size:
                pretrained_weights = pretrained_weights[:self.vocab_size]
            else:
                # Pad with random initialization
                padding = np.random.randn(
                    self.vocab_size - pretrained_weights.shape[0],
                    self.embedding_dim
                ) * 0.01
                pretrained_weights = np.vstack([pretrained_weights, padding])
        
        # Convert to tensor and load
        pretrained_tensor = torch.from_numpy(pretrained_weights).float()
        self.embedding.weight.data.copy_(pretrained_tensor)
        
        # Ensure padding is zero
        self.embedding.weight.data[self.padding_idx].fill_(0)
        
        logger.info(f"Loaded pretrained weights: {pretrained_weights.shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input codes.
        
        Args:
            x: Code indices [batch_size, seq_len] or [batch_size, seq_len, codes_per_visit]
        
        Returns:
            Embeddings [batch_size, seq_len, embed_dim] or 
                       [batch_size, seq_len, codes_per_visit, embed_dim]
        """
        return self.embedding(x)
    
    def unfreeze(self):
        """Allow fine-tuning of pretrained embeddings."""
        self.embedding.weight.requires_grad = True
        self.freeze = False
        logger.info("Unfroze pretrained embedding weights")
    
    @classmethod
    def from_file(
        cls,
        embedding_path: Path,
        vocab: Dict[str, int],
        embedding_dim: int,
        code_to_embedding_idx: Optional[Dict[str, int]] = None,
        freeze: bool = True,
    ) -> 'PretrainedEmbedding':
        """
        Load pretrained embeddings from file.
        
        Args:
            embedding_path: Path to embedding file (.pkl, .npy, or .txt)
            vocab: Target vocabulary (code -> index)
            embedding_dim: Expected embedding dimension
            code_to_embedding_idx: Mapping from codes to pretrained embedding indices
            freeze: Whether to freeze weights
        
        Returns:
            PretrainedEmbedding instance
        """
        embedding_path = Path(embedding_path)
        
        if not embedding_path.exists():
            logger.warning(f"Embedding file not found: {embedding_path}")
            logger.warning("Using random initialization instead")
            return cls(vocab, embedding_dim, pretrained_weights=None, freeze=False)
        
        # Load embeddings based on file type
        if embedding_path.suffix == '.pkl':
            with open(embedding_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            if isinstance(embedding_data, dict):
                # Format: {code: embedding_vector}
                pretrained_weights = cls._build_weight_matrix_from_dict(
                    embedding_data, vocab, embedding_dim
                )
            elif isinstance(embedding_data, np.ndarray):
                # Format: embedding matrix
                pretrained_weights = embedding_data
            else:
                raise ValueError(f"Unsupported embedding format: {type(embedding_data)}")
        
        elif embedding_path.suffix == '.npy':
            pretrained_weights = np.load(embedding_path)
        
        elif embedding_path.suffix == '.txt':
            pretrained_weights = cls._load_from_text(
                embedding_path, vocab, embedding_dim, code_to_embedding_idx
            )
        
        else:
            raise ValueError(f"Unsupported file format: {embedding_path.suffix}")
        
        return cls(vocab, embedding_dim, pretrained_weights, freeze)
    
    @staticmethod
    def _build_weight_matrix_from_dict(
        embedding_dict: Dict[str, np.ndarray],
        vocab: Dict[str, int],
        embedding_dim: int
    ) -> np.ndarray:
        """
        Build weight matrix from code -> embedding dictionary.
        
        Args:
            embedding_dict: {code: embedding_vector}
            vocab: Target vocabulary
            embedding_dim: Embedding dimension
        
        Returns:
            Weight matrix [vocab_size, embedding_dim]
        """
        vocab_size = len(vocab)
        weights = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        found = 0
        for code, idx in vocab.items():
            if code in embedding_dict:
                weights[idx] = embedding_dict[code]
                found += 1
        
        coverage = found / vocab_size * 100
        logger.info(f"Pretrained embedding coverage: {coverage:.1f}% ({found}/{vocab_size})")
        
        return weights
    
    @staticmethod
    def _load_from_text(
        embedding_path: Path,
        vocab: Dict[str, int],
        embedding_dim: int,
        code_to_embedding_idx: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        """
        Load embeddings from text file (word2vec format).
        
        Format:
            code1 0.1 0.2 0.3 ...
            code2 0.4 0.5 0.6 ...
        
        Args:
            embedding_path: Path to text file
            vocab: Target vocabulary
            embedding_dim: Embedding dimension
            code_to_embedding_idx: Optional code mapping
        
        Returns:
            Weight matrix [vocab_size, embedding_dim]
        """
        embedding_dict = {}
        
        with open(embedding_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                
                code = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                embedding_dict[code] = vector
        
        return PretrainedEmbedding._build_weight_matrix_from_dict(
            embedding_dict, vocab, embedding_dim
        )


class Med2VecEmbedding(PretrainedEmbedding):
    """
    Med2Vec pretrained embeddings.
    
    Med2Vec learns embeddings for medical codes using skip-gram with
    visit-level context. Codes that co-occur in visits have similar embeddings.
    
    Reference:
        Choi et al. (2016). "Multi-layer Representation Learning for Medical Concepts"
        https://arxiv.org/abs/1602.05568
    
    Example:
        >>> embedding = Med2VecEmbedding.from_med2vec_checkpoint(
        ...     checkpoint_path='med2vec_model.pkl',
        ...     vocab=vocab
        ... )
    """
    
    @classmethod
    def from_med2vec_checkpoint(
        cls,
        checkpoint_path: Path,
        vocab: Dict[str, int],
        freeze: bool = True
    ) -> 'Med2VecEmbedding':
        """
        Load Med2Vec embeddings from checkpoint.
        
        Args:
            checkpoint_path: Path to Med2Vec checkpoint
            vocab: Target vocabulary
            freeze: Whether to freeze weights
        
        Returns:
            Med2VecEmbedding instance
        """
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Extract code embeddings from checkpoint
        # Format depends on Med2Vec implementation
        if 'code_embeddings' in checkpoint:
            code_embeddings = checkpoint['code_embeddings']
            code_to_idx = checkpoint.get('code_to_idx', {})
        else:
            raise ValueError("Invalid Med2Vec checkpoint format")
        
        embedding_dim = code_embeddings.shape[1]
        
        # Build weight matrix
        weights = np.random.randn(len(vocab), embedding_dim) * 0.01
        found = 0
        
        for code, vocab_idx in vocab.items():
            if code in code_to_idx:
                emb_idx = code_to_idx[code]
                weights[vocab_idx] = code_embeddings[emb_idx]
                found += 1
        
        coverage = found / len(vocab) * 100
        logger.info(f"Med2Vec coverage: {coverage:.1f}% ({found}/{len(vocab)})")
        
        return cls(vocab, embedding_dim, weights, freeze)


class CUI2VecEmbedding(PretrainedEmbedding):
    """
    CUI2Vec pretrained embeddings for UMLS concepts.
    
    CUI2Vec learns embeddings for UMLS Concept Unique Identifiers (CUIs)
    using clinical notes. Requires mapping ICD/CPT codes to CUIs.
    
    Reference:
        Beam et al. (2018). "Clinical Concept Embeddings Learned from Massive Sources of Multimodal Medical Data"
        https://arxiv.org/abs/1804.01486
    """
    
    pass  # Similar implementation to Med2Vec


class ClinicalBERTEmbedding(PretrainedEmbedding):
    """
    Clinical BERT embeddings for medical codes.
    
    Uses contextualized embeddings from clinical BERT models
    (BioBERT, ClinicalBERT, etc.) for medical codes.
    
    Reference:
        Alsentzer et al. (2019). "Publicly Available Clinical BERT Embeddings"
    """
    
    pass  # Implementation for BERT-based embeddings
