"""
Embedding modules for medical codes.

This package provides:
- Pretrained medical code embeddings (Med2Vec, CUI2Vec, Clinical BERT)
- Custom embedding layers with medical code-specific features
- Embedding utilities and transformations
"""

from .pretrained import (
    PretrainedEmbedding,
    Med2VecEmbedding,
    CUI2VecEmbedding,
    ClinicalBERTEmbedding,
)

__all__ = [
    'PretrainedEmbedding',
    'Med2VecEmbedding',
    'CUI2VecEmbedding',
    'ClinicalBERTEmbedding',
]

"""
Medical code embeddings (Med2Vec, graph-based, etc.).
"""

# from ehrsequencing.embeddings.med2vec import Med2Vec
# from ehrsequencing.embeddings.graph import GraphEmbeddings
