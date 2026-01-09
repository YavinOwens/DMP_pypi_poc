"""
Abstract base classes for vector stores and embedding providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    """
    Abstract interface for vector store backends
    
    All vector store implementations must inherit from this class
    and implement all abstract methods.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store (create tables/collections if needed)"""
        pass
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add vectors to the store
        
        Args:
            vectors: List of embedding vectors
            texts: List of text content
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (auto-generated if not provided)
            
        Returns:
            List of IDs for the added vectors
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of results with 'text', 'metadata', 'score', 'id' keys
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of vectors in the store"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all vectors from the store"""
        pass


class EmbeddingProvider(ABC):
    """
    Abstract interface for embedding providers
    
    All embedding provider implementations must inherit from this class
    and implement all abstract methods.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the embedding model"""
        pass
