"""
FAISS vector store implementation

Requires: pip install faiss-cpu (or faiss-gpu for GPU support)
"""

import logging
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import VectorStore
from .local_storage import LocalStorageManager

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation with local storage"""
    
    def __init__(
        self,
        index_name: str = "knowledge_base",
        dimension: int = 768,
        storage_path: Optional[Path] = None,
        storage_manager: Optional[LocalStorageManager] = None
    ):
        """
        Initialize FAISS vector store
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            storage_path: Path to store FAISS index
            storage_manager: LocalStorageManager instance
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu or faiss-gpu required for FAISS backend. "
                "Install with: pip install faiss-cpu"
            )
        
        self.index_name = index_name
        self.dimension = dimension
        
        # Use storage manager if provided
        if storage_manager:
            self.storage_manager = storage_manager
            storage_path = storage_manager.get_vector_store_path("faiss", index_name)
        elif storage_path:
            self.storage_manager = LocalStorageManager(storage_path.parent.parent)
        else:
            self.storage_manager = LocalStorageManager()
            storage_path = self.storage_manager.get_vector_store_path("faiss", index_name)
        
        self.index_path = storage_path / f"{index_name}.index"
        self.metadata_db_path = storage_path / f"{index_name}.db"
        
        # Initialize FAISS index
        self.index = None
        self.metadata_db = None
        self._id_to_index = {}  # Map vector IDs to FAISS index positions
        self._index_to_id = {}  # Reverse mapping
        self._initialized = False
    
    def _init_metadata_db(self) -> None:
        """Initialize SQLite database for metadata"""
        self.metadata_db = sqlite3.connect(self.metadata_db_path)
        cursor = self.metadata_db.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                text TEXT,
                metadata_json TEXT,
                faiss_index INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_faiss_index 
            ON vectors(faiss_index)
        """)
        
        self.metadata_db.commit()
    
    def initialize(self) -> None:
        """Create or load FAISS index"""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_metadata_db()
            
            if self.index_path.exists():
                # Load existing index
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"✓ Loaded existing FAISS index: {self.index_name}")
                
                # Load metadata mappings
                cursor = self.metadata_db.cursor()
                cursor.execute("SELECT id, faiss_index FROM vectors")
                for row in cursor.fetchall():
                    vector_id, faiss_idx = row
                    self._id_to_index[vector_id] = faiss_idx
                    self._index_to_id[faiss_idx] = vector_id
            else:
                # Create new index (L2 distance, inner product also available)
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"✓ Created new FAISS index: {self.index_name}")
            
            # Save config to local storage
            self.storage_manager.save_vector_store_config(
                backend="faiss",
                collection_name=self.index_name,
                dimension=self.dimension
            )
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")
            raise
    
    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to FAISS"""
        if not self._initialized:
            self.initialize()
        
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(vectors))]
        
        if not (len(vectors) == len(texts) == len(metadatas) == len(ids)):
            raise ValueError("All input lists must have the same length")
        
        try:
            # Convert to numpy array
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # Get current index size (for mapping)
            current_size = self.index.ntotal
            
            # Add to FAISS index
            self.index.add(vectors_array)
            
            # Store metadata in SQLite
            cursor = self.metadata_db.cursor()
            import json
            
            for i, (vector_id, text, metadata) in enumerate(zip(ids, texts, metadatas)):
                faiss_idx = current_size + i
                self._id_to_index[vector_id] = faiss_idx
                self._index_to_id[faiss_idx] = vector_id
                
                metadata_json = json.dumps(metadata)
                cursor.execute("""
                    INSERT OR REPLACE INTO vectors (id, text, metadata_json, faiss_index)
                    VALUES (?, ?, ?, ?)
                """, (vector_id, text, metadata_json, faiss_idx))
                
                # Save to local storage manager
                self.storage_manager.save_document_metadata(
                    document_id=vector_id,
                    source_path=metadata.get("source", ""),
                    chunk_index=metadata.get("chunk_index", 0),
                    backend="faiss",
                    vector_id=vector_id,
                    metadata=metadata
                )
            
            self.metadata_db.commit()
            self._save_index()
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS: {e}")
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
            # Convert query to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Search in FAISS
            distances, indices = self.index.search(query_array, top_k)
            
            formatted_results = []
            cursor = self.metadata_db.cursor()
            import json
            
            for i, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
                if faiss_idx == -1:  # FAISS returns -1 for empty results
                    continue
                
                vector_id = self._index_to_id.get(faiss_idx)
                if not vector_id:
                    continue
                
                # Get metadata from SQLite
                cursor.execute(
                    "SELECT text, metadata_json FROM vectors WHERE id = ?",
                    (vector_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    text, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    # Apply filter if provided
                    if filter:
                        if not all(metadata.get(k) == v for k, v in filter.items()):
                            continue
                    
                    formatted_results.append({
                        'id': vector_id,
                        'text': text or '',
                        'metadata': metadata,
                        'score': float(1.0 / (1.0 + distance))  # Convert L2 distance to similarity score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (FAISS doesn't support deletion, so we mark as deleted)"""
        if not ids:
            return True
        
        try:
            # FAISS doesn't support deletion, so we remove from metadata
            cursor = self.metadata_db.cursor()
            for vector_id in ids:
                cursor.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))
                if vector_id in self._id_to_index:
                    faiss_idx = self._id_to_index[vector_id]
                    del self._id_to_index[vector_id]
                    if faiss_idx in self._index_to_id:
                        del self._index_to_id[faiss_idx]
            
            self.metadata_db.commit()
            # Note: FAISS index itself isn't modified (would require rebuilding)
            logger.warning("FAISS doesn't support true deletion - vectors marked as deleted in metadata only")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from FAISS: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of vectors"""
        try:
            cursor = self.metadata_db.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting FAISS vectors: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            # Delete index file
            if self.index_path.exists():
                self.index_path.unlink()
            
            # Clear metadata
            cursor = self.metadata_db.cursor()
            cursor.execute("DELETE FROM vectors")
            self.metadata_db.commit()
            
            # Reinitialize
            self._id_to_index = {}
            self._index_to_id = {}
            self.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing FAISS: {e}")
            return False
