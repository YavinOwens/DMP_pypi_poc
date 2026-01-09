"""
Qdrant vector store implementation

Requires: pip install qdrant-client
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import VectorStore
from .local_storage import LocalStorageManager

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation with local storage"""
    
    def __init__(
        self,
        collection_name: str = "knowledge_base",
        dimension: int = 768,
        storage_path: Optional[Path] = None,
        storage_manager: Optional[LocalStorageManager] = None
    ):
        """
        Initialize Qdrant vector store
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            storage_path: Path to store Qdrant data (local mode)
            storage_manager: LocalStorageManager instance
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client required for Qdrant backend. "
                "Install with: pip install qdrant-client"
            )
        
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Use storage manager if provided, otherwise create default
        if storage_manager:
            self.storage_manager = storage_manager
            storage_path = storage_manager.get_vector_store_path("qdrant", collection_name)
        elif storage_path:
            self.storage_manager = LocalStorageManager(storage_path.parent.parent)
        else:
            self.storage_manager = LocalStorageManager()
            storage_path = self.storage_manager.get_vector_store_path("qdrant", collection_name)
        
        # Initialize Qdrant client in local mode
        self.client = QdrantClient(path=str(storage_path))
        self._initialized = False
    
    def initialize(self) -> None:
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"✓ Qdrant collection exists: {self.collection_name}")
            
            # Save config to local storage
            self.storage_manager.save_vector_store_config(
                backend="qdrant",
                collection_name=self.collection_name,
                dimension=self.dimension
            )
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to Qdrant"""
        if not self._initialized:
            self.initialize()
        
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(vectors))]
        
        if not (len(vectors) == len(texts) == len(metadatas) == len(ids)):
            raise ValueError("All input lists must have the same length")
        
        # Prepare points
        points = []
        for vector, text, metadata, point_id in zip(vectors, texts, metadatas, ids):
            # Add text to metadata
            full_metadata = {**metadata, "text": text}
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload=full_metadata
            )
            points.append(point)
            
            # Save metadata to local storage
            self.storage_manager.save_document_metadata(
                document_id=point_id,
                source_path=metadata.get("source", ""),
                chunk_index=metadata.get("chunk_index", 0),
                backend="qdrant",
                vector_id=point_id,
                metadata=full_metadata
            )
        
        # Upsert points
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return ids
        except Exception as e:
            logger.error(f"Error adding vectors to Qdrant: {e}")
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
            # Convert filter to Qdrant format if provided
            qdrant_filter = None
            if filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                conditions = []
                for key, value in filter.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': str(result.id),
                    'text': result.payload.get('text', ''),
                    'metadata': {k: v for k, v in result.payload.items() if k != 'text'},
                    'score': float(result.score)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        if not ids:
            return True
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            return True
        except Exception as e:
            logger.error(f"Error deleting from Qdrant: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of vectors"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Error counting Qdrant vectors: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            self.client.delete_collection(self.collection_name)
            self.initialize()  # Recreate empty collection
            return True
        except Exception as e:
            logger.error(f"Error clearing Qdrant: {e}")
            return False
