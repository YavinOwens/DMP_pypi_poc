"""
Factory for creating vector stores and embedding providers
"""

import logging
from typing import Optional, Any

from .base import VectorStore, EmbeddingProvider
from .snowflake import SnowflakeVectorStore, SnowflakeEmbeddingProvider
from .local_storage import LocalStorageManager
from ..config import RAGConfig

logger = logging.getLogger(__name__)


def create_vector_store(
    backend: Optional[str] = None,
    config: Optional[RAGConfig] = None,
    session: Optional[Any] = None,
    **kwargs
) -> VectorStore:
    """
    Factory function to create a vector store instance
    
    Args:
        backend: Backend name (snowflake, qdrant, chromadb, faiss)
                If None, uses config or defaults to snowflake
        config: RAGConfig instance (optional)
        session: Snowflake session (required for snowflake backend)
        **kwargs: Additional backend-specific arguments
        
    Returns:
        VectorStore instance
    """
    if config is None:
        config = RAGConfig()
    
    backend = backend or config.get_backend()
    backend_config = config.get_backend_config(backend)
    
    if backend == "snowflake":
        if not session:
            raise ValueError("Snowflake session required for snowflake backend")
        
        return SnowflakeVectorStore(
            session=session,
            table_name=backend_config.get("table", "VECTOR_STORE"),
            schema=backend_config.get("schema", "PUBLIC"),
            database=backend_config.get("database"),
            dimension=config.get_embedding_config().get("dimension", 1024),
            **kwargs
        )
    
    elif backend == "qdrant":
        from .qdrant_backend import QdrantVectorStore
        
        storage_manager = LocalStorageManager(config.get_local_storage_path())
        return QdrantVectorStore(
            collection_name=backend_config.get("collection_name", "knowledge_base"),
            dimension=config.get_embedding_config().get("dimension", 768),
            storage_manager=storage_manager,
            **kwargs
        )
    
    elif backend == "chromadb":
        from .chromadb_backend import ChromaDBVectorStore
        
        storage_manager = LocalStorageManager(config.get_local_storage_path())
        return ChromaDBVectorStore(
            collection_name=backend_config.get("collection_name", "knowledge_base"),
            dimension=config.get_embedding_config().get("dimension", 768),
            storage_manager=storage_manager,
            **kwargs
        )
    
    elif backend == "faiss":
        from .faiss_backend import FAISSVectorStore
        
        storage_manager = LocalStorageManager(config.get_local_storage_path())
        return FAISSVectorStore(
            index_name=backend_config.get("index_name", "knowledge_base"),
            dimension=config.get_embedding_config().get("dimension", 768),
            storage_manager=storage_manager,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: snowflake, qdrant, chromadb, faiss")


def create_embedding_provider(
    provider: Optional[str] = None,
    config: Optional[RAGConfig] = None,
    session: Optional[Any] = None,
    **kwargs
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider instance
    
    Args:
        provider: Provider name (snowflake, openai, huggingface, local)
                 If None, uses config or defaults to snowflake
        config: RAGConfig instance (optional)
        session: Snowflake session (required for snowflake provider)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        EmbeddingProvider instance
    """
    if config is None:
        config = RAGConfig()
    
    embedding_config = config.get_embedding_config()
    provider = provider or embedding_config.get("provider", "snowflake")
    model = embedding_config.get("model", "snowflake-arctic-embed-l-v2.0")
    
    if provider == "snowflake":
        if not session:
            raise ValueError("Snowflake session required for snowflake embedding provider")
        
        return SnowflakeEmbeddingProvider(
            session=session,
            model=model,
            **kwargs
        )
    
    elif provider == "openai":
        # TODO: Implement OpenAI embedding provider
        raise NotImplementedError("OpenAI embedding provider not yet implemented")
    
    elif provider == "huggingface" or provider == "local":
        # TODO: Implement local/HuggingFace embedding provider
        raise NotImplementedError("Local/HuggingFace embedding provider not yet implemented")
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}. Supported: snowflake, openai, huggingface, local")
