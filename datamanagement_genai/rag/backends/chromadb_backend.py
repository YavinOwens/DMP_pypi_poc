"""
ChromaDB vector store implementation

Requires: pip install chromadb
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import VectorStore
from .local_storage import LocalStorageManager

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation with local storage"""
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        dimension: int = 768,
        storage_path: Optional[Path] = None,
        storage_manager: Optional[LocalStorageManager] = None
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            storage_path: Path to store ChromaDB data
            storage_manager: LocalStorageManager instance
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb required for ChromaDB backend. "
                "Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Use storage manager if provided
        if storage_manager:
            self.storage_manager = storage_manager
            storage_path = storage_manager.get_vector_store_path("chromadb", collection_name)
        elif storage_path:
            self.storage_manager = LocalStorageManager(storage_path.parent.parent)
        else:
            self.storage_manager = LocalStorageManager()
            storage_path = self.storage_manager.get_vector_store_path("chromadb", collection_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Create or get collection"""
        try:
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimension": self.dimension}
            )
            
            logger.info(f"✓ ChromaDB collection ready: {self.collection_name}")
            
            # Save config to local storage
            self.storage_manager.save_vector_store_config(
                backend="chromadb",
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to ChromaDB"""
        if not self._initialized:
            self.initialize()
        
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(vectors))]
        
        if not (len(vectors) == len(texts) == len(metadatas) == len(ids)):
            raise ValueError("All input lists must have the same length")
        
        try:
            # ChromaDB expects documents, embeddings, metadatas, ids
            self.collection.add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Save metadata to local storage
            for vector_id, metadata in zip(ids, metadatas):
                self.storage_manager.save_document_metadata(
                    document_id=vector_id,
                    source_path=metadata.get("source", ""),
                    chunk_index=metadata.get("chunk_index", 0),
                    backend="chromadb",
                    vector_id=vector_id,
                    metadata=metadata
                )
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding vectors to ChromaDB: {e}")
            return []
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if not self._initialized:
            self.initialize()
        
        try:
            # Convert filter to ChromaDB format
            where = filter if filter else None
            
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where
            )
            
            formatted_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i] if results['documents'] else '',
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0  # Convert distance to similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        if not ids:
            return True
        
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting from ChromaDB: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of vectors"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting ChromaDB vectors: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.initialize()  # Recreate empty collection
            return True
        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")
            return False
