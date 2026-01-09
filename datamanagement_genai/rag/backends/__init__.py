"""
Vector Store and Embedding Provider Backends

This module provides abstract interfaces and implementations for various
vector database backends and embedding providers.
"""

from .base import VectorStore, EmbeddingProvider
from .snowflake import SnowflakeVectorStore, SnowflakeEmbeddingProvider
from .local_storage import LocalStorageManager

__all__ = [
    'VectorStore',
    'EmbeddingProvider',
    'SnowflakeVectorStore',
    'SnowflakeEmbeddingProvider',
    'LocalStorageManager',
]

# Lazy imports for optional backends
try:
    from .qdrant_backend import QdrantVectorStore
    __all__.append('QdrantVectorStore')
except ImportError:
    QdrantVectorStore = None

try:
    from .chromadb_backend import ChromaDBVectorStore
    __all__.append('ChromaDBVectorStore')
except ImportError:
    ChromaDBVectorStore = None

try:
    from .faiss_backend import FAISSVectorStore
    __all__.append('FAISSVectorStore')
except ImportError:
    FAISSVectorStore = None
