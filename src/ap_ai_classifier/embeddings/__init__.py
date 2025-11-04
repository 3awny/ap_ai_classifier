"""
Embeddings module for AP AI Classifier.
"""
from .openai_embedding import OpenAIEmbeddingModel, create_embedding_model
from .embedding_config import EmbeddingConfig, DEFAULT_CONFIG

__all__ = [
    'OpenAIEmbeddingModel',
    'create_embedding_model',
    'EmbeddingConfig',
    'DEFAULT_CONFIG',
]

# Backward compatibility aliases
MultilingualEmbeddingModel = OpenAIEmbeddingModel
EmbeddingModelConfig = EmbeddingConfig

