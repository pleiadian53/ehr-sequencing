"""
Utilities for loading and using pre-trained medical code embeddings.

Supports:
- Med2Vec embeddings (from Phase 2)
- Word2Vec embeddings
- Custom pre-trained embeddings
- Embedding initialization from pre-trained weights
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import pickle


def load_med2vec_embeddings(
    embedding_path: Union[str, Path],
    vocab_size: int,
    embedding_dim: int
) -> torch.Tensor:
    """
    Load Med2Vec embeddings from saved checkpoint.
    
    Args:
        embedding_path: Path to Med2Vec embedding file (.pt or .pkl)
        vocab_size: Expected vocabulary size
        embedding_dim: Expected embedding dimension
    
    Returns:
        Embedding tensor [vocab_size, embedding_dim]
    
    Example:
        >>> embeddings = load_med2vec_embeddings('med2vec_embeddings.pt', 1000, 128)
        >>> print(embeddings.shape)
        torch.Size([1000, 128])
    """
    embedding_path = Path(embedding_path)
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    
    # Load embeddings
    if embedding_path.suffix == '.pt':
        embeddings = torch.load(embedding_path, map_location='cpu')
    elif embedding_path.suffix == '.pkl':
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
    else:
        raise ValueError(f"Unsupported file format: {embedding_path.suffix}")
    
    # Validate shape
    if embeddings.shape != (vocab_size, embedding_dim):
        raise ValueError(
            f"Embedding shape mismatch: expected ({vocab_size}, {embedding_dim}), "
            f"got {embeddings.shape}"
        )
    
    return embeddings


def load_word2vec_embeddings(
    embedding_path: Union[str, Path],
    vocab_mapping: Dict[int, str],
    embedding_dim: int
) -> torch.Tensor:
    """
    Load Word2Vec embeddings and map to medical code vocabulary.
    
    Args:
        embedding_path: Path to Word2Vec model file
        vocab_mapping: Dictionary mapping code IDs to code strings
        embedding_dim: Embedding dimension
    
    Returns:
        Embedding tensor [vocab_size, embedding_dim]
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("gensim is required for Word2Vec embeddings. Install with: pip install gensim")
    
    embedding_path = Path(embedding_path)
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"Word2Vec model not found: {embedding_path}")
    
    # Load Word2Vec model
    model = Word2Vec.load(str(embedding_path))
    
    # Initialize embeddings with random values
    vocab_size = len(vocab_mapping)
    embeddings = torch.randn(vocab_size, embedding_dim) * 0.01
    
    # Fill in embeddings for codes that exist in Word2Vec
    found_count = 0
    for code_id, code_str in vocab_mapping.items():
        if code_str in model.wv:
            embeddings[code_id] = torch.from_numpy(model.wv[code_str]).float()
            found_count += 1
    
    print(f"Loaded Word2Vec embeddings: {found_count}/{vocab_size} codes found")
    
    return embeddings


def initialize_embedding_layer(
    embedding_layer: nn.Embedding,
    pretrained_embeddings: torch.Tensor,
    freeze: bool = True
) -> nn.Embedding:
    """
    Initialize an embedding layer with pre-trained weights.
    
    Args:
        embedding_layer: PyTorch embedding layer to initialize
        pretrained_embeddings: Pre-trained embedding tensor
        freeze: Whether to freeze the embedding layer
    
    Returns:
        Initialized embedding layer
    
    Example:
        >>> embedding = nn.Embedding(1000, 128)
        >>> pretrained = load_med2vec_embeddings('embeddings.pt', 1000, 128)
        >>> embedding = initialize_embedding_layer(embedding, pretrained, freeze=True)
    """
    # Validate shapes match
    if embedding_layer.weight.shape != pretrained_embeddings.shape:
        raise ValueError(
            f"Shape mismatch: embedding layer has shape {embedding_layer.weight.shape}, "
            f"but pretrained embeddings have shape {pretrained_embeddings.shape}"
        )
    
    # Copy weights
    with torch.no_grad():
        embedding_layer.weight.copy_(pretrained_embeddings)
    
    # Freeze if requested
    if freeze:
        embedding_layer.weight.requires_grad = False
        print(f"✅ Initialized embedding layer with pre-trained weights (frozen)")
    else:
        embedding_layer.weight.requires_grad = True
        print(f"✅ Initialized embedding layer with pre-trained weights (trainable)")
    
    return embedding_layer


def save_embeddings(
    embeddings: torch.Tensor,
    save_path: Union[str, Path],
    metadata: Optional[Dict] = None
):
    """
    Save embeddings to disk with optional metadata.
    
    Args:
        embeddings: Embedding tensor to save
        save_path: Path to save embeddings
        metadata: Optional metadata dictionary
    
    Example:
        >>> embeddings = model.embeddings.code_embedding.weight.data
        >>> save_embeddings(
        ...     embeddings,
        ...     'behrt_embeddings.pt',
        ...     metadata={'vocab_size': 1000, 'embedding_dim': 128}
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'embeddings': embeddings.cpu(),
        'shape': embeddings.shape,
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    print(f"💾 Saved embeddings to {save_path}")
    print(f"   Shape: {embeddings.shape}")
    if metadata:
        print(f"   Metadata: {metadata}")


def load_embeddings(
    load_path: Union[str, Path]
) -> tuple[torch.Tensor, Optional[Dict]]:
    """
    Load embeddings from disk.
    
    Args:
        load_path: Path to embedding file
    
    Returns:
        Tuple of (embeddings, metadata)
    
    Example:
        >>> embeddings, metadata = load_embeddings('behrt_embeddings.pt')
        >>> print(embeddings.shape)
        torch.Size([1000, 128])
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {load_path}")
    
    save_dict = torch.load(load_path, map_location='cpu')
    
    embeddings = save_dict['embeddings']
    metadata = save_dict.get('metadata', None)
    
    print(f"📂 Loaded embeddings from {load_path}")
    print(f"   Shape: {embeddings.shape}")
    if metadata:
        print(f"   Metadata: {metadata}")
    
    return embeddings, metadata


def create_random_embeddings(
    vocab_size: int,
    embedding_dim: int,
    std: float = 0.02,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create random embeddings for testing.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        std: Standard deviation for initialization
        seed: Random seed
    
    Returns:
        Random embedding tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    embeddings = torch.randn(vocab_size, embedding_dim) * std
    
    # Set padding embedding to zero
    embeddings[0] = 0.0
    
    return embeddings


def get_embedding_statistics(embeddings: torch.Tensor) -> Dict:
    """
    Compute statistics about embeddings.
    
    Args:
        embeddings: Embedding tensor
    
    Returns:
        Dictionary of statistics
    """
    return {
        'shape': embeddings.shape,
        'mean': embeddings.mean().item(),
        'std': embeddings.std().item(),
        'min': embeddings.min().item(),
        'max': embeddings.max().item(),
        'norm_mean': embeddings.norm(dim=1).mean().item(),
        'norm_std': embeddings.norm(dim=1).std().item(),
    }


def print_embedding_statistics(embeddings: torch.Tensor, name: str = "Embeddings"):
    """Print embedding statistics in a readable format."""
    stats = get_embedding_statistics(embeddings)
    
    print(f"\n📊 {name} Statistics:")
    print(f"   Shape: {stats['shape']}")
    print(f"   Value range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"   Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"   Norm mean: {stats['norm_mean']:.4f}, Norm std: {stats['norm_std']:.4f}")
