"""
Local storage manager for vector stores and metadata

Manages SQLite database for configuration and metadata,
and provides utilities for local file storage.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalStorageManager:
    """
    Manages local storage for vector stores and metadata
    
    Uses SQLite for configuration and metadata storage,
    and manages local directories for vector store data.
    """
    
    def __init__(self, storage_root: Optional[Path] = None):
        """
        Initialize local storage manager
        
        Args:
            storage_root: Root directory for local storage.
                        If None, uses .datamanagement_genai in current directory
        """
        if storage_root is None:
            # Use current working directory or package root
            storage_root = Path.cwd() / ".datamanagement_genai"
        
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for config and metadata
        self.db_path = self.storage_root / "config.db"
        self.vector_stores_dir = self.storage_root / "vector_stores"
        self.vector_stores_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vector store configuration table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_store_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backend TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(backend, collection_name)
            )
        """)
        
        # Document metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE NOT NULL,
                source_path TEXT,
                chunk_index INTEGER,
                backend TEXT NOT NULL,
                vector_id TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_metadata_backend 
            ON document_metadata(backend)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_metadata_source 
            ON document_metadata(source_path)
        """)
        
        conn.commit()
        conn.close()
        logger.debug(f"Initialized local storage database: {self.db_path}")
    
    def get_vector_store_path(self, backend: str, collection_name: str) -> Path:
        """
        Get storage path for a vector store backend
        
        Args:
            backend: Backend name (qdrant, chromadb, faiss, etc.)
            collection_name: Collection/table name
            
        Returns:
            Path to store vector data
        """
        backend_dir = self.vector_stores_dir / backend
        backend_dir.mkdir(exist_ok=True)
        return backend_dir / collection_name
    
    def save_vector_store_config(
        self,
        backend: str,
        collection_name: str,
        dimension: int,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save vector store configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        config_json = json.dumps(config) if config else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO vector_store_config 
            (backend, collection_name, dimension, config_json, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (backend, collection_name, dimension, config_json))
        
        conn.commit()
        conn.close()
    
    def get_vector_store_config(
        self,
        backend: str,
        collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get vector store configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT dimension, config_json 
            FROM vector_store_config 
            WHERE backend = ? AND collection_name = ?
        """, (backend, collection_name))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            dimension, config_json = row
            config = json.loads(config_json) if config_json else {}
            config['dimension'] = dimension
            return config
        return None
    
    def save_document_metadata(
        self,
        document_id: str,
        source_path: str,
        chunk_index: int,
        backend: str,
        vector_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Save document metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata)
        
        cursor.execute("""
            INSERT OR REPLACE INTO document_metadata 
            (document_id, source_path, chunk_index, backend, vector_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (document_id, source_path, chunk_index, backend, vector_id, metadata_json))
        
        conn.commit()
        conn.close()
    
    def get_document_metadata(
        self,
        document_id: Optional[str] = None,
        source_path: Optional[str] = None,
        backend: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get document metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM document_metadata WHERE 1=1"
        params = []
        
        if document_id:
            query += " AND document_id = ?"
            params.append(document_id)
        if source_path:
            query += " AND source_path = ?"
            params.append(source_path)
        if backend:
            query += " AND backend = ?"
            params.append(backend)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        results = []
        for row in rows:
            result = dict(zip(columns, row))
            if result.get('metadata_json'):
                result['metadata'] = json.loads(result['metadata_json'])
            results.append(result)
        
        return results
    
    def list_vector_stores(self) -> List[Dict[str, Any]]:
        """List all configured vector stores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT backend, collection_name, dimension, created_at, updated_at
            FROM vector_store_config
            ORDER BY backend, collection_name
        """)
        
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def cleanup(self, backend: Optional[str] = None) -> None:
        """
        Clean up local storage
        
        Args:
            backend: If specified, only clean up this backend's data
        """
        if backend:
            backend_dir = self.vector_stores_dir / backend
            if backend_dir.exists():
                import shutil
                shutil.rmtree(backend_dir)
                logger.info(f"Cleaned up {backend} storage")
        else:
            # Clean up all vector stores (but keep config.db)
            if self.vector_stores_dir.exists():
                import shutil
                shutil.rmtree(self.vector_stores_dir)
                self.vector_stores_dir.mkdir(exist_ok=True)
                logger.info("Cleaned up all vector store data")
