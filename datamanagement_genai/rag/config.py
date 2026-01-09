"""
Configuration management for RAG system backends
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

logger = logging.getLogger(__name__)


class RAGConfig:
    """Configuration manager for RAG system backends"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize RAG configuration
        
        Args:
            config_path: Path to config file (TOML or JSON)
                        If None, searches for config files
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations"""
        search_paths = [
            Path.cwd() / "config.toml",
            Path.cwd() / ".datamanagement_genai" / "config.toml",
            Path.cwd() / ".streamlit" / "secrets.toml",
            Path.home() / ".datamanagement_genai" / "config.toml",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config = {
            "backend": os.getenv("RAG_BACKEND", "snowflake"),
            "local_storage_path": os.getenv("RAG_LOCAL_STORAGE", ".datamanagement_genai"),
            "embeddings": {
                "provider": os.getenv("RAG_EMBEDDING_PROVIDER", "snowflake"),
                "model": os.getenv("RAG_EMBEDDING_MODEL", "snowflake-arctic-embed-l-v2.0"),
                "dimension": int(os.getenv("RAG_EMBEDDING_DIMENSION", "1024")),
            },
            "snowflake": {
                "database": os.getenv("SNOWFLAKE_RAG_DATABASE", "RAG_KNOWLEDGE_BASE"),
                "schema": os.getenv("SNOWFLAKE_RAG_SCHEMA", "PUBLIC"),
                "table": os.getenv("SNOWFLAKE_RAG_TABLE", "VECTOR_STORE"),
            },
            "qdrant": {
                "collection_name": os.getenv("QDRANT_COLLECTION", "knowledge_base"),
            },
            "chromadb": {
                "collection_name": os.getenv("CHROMADB_COLLECTION", "knowledge_base"),
            },
            "faiss": {
                "index_name": os.getenv("FAISS_INDEX", "knowledge_base"),
            },
        }
        
        # Load from file if exists
        if self.config_path and self.config_path.exists():
            try:
                if self.config_path.suffix == ".toml" and tomllib:
                    with open(self.config_path, "rb") as f:
                        file_config = tomllib.load(f)
                    
                    # Extract RAG config
                    if "rag" in file_config:
                        rag_config = file_config["rag"]
                        config.update({
                            "backend": rag_config.get("backend", config["backend"]),
                            "local_storage_path": rag_config.get("local_storage_path", config["local_storage_path"]),
                        })
                        
                        # Update backend-specific configs
                        for backend in ["snowflake", "qdrant", "chromadb", "faiss"]:
                            if backend in rag_config:
                                config[backend].update(rag_config[backend])
                        
                        # Update embeddings config
                        if "embeddings" in rag_config:
                            config["embeddings"].update(rag_config["embeddings"])
                
                elif self.config_path.suffix == ".json":
                    with open(self.config_path, "r") as f:
                        file_config = json.load(f)
                    
                    if "rag" in file_config:
                        # Similar update logic for JSON
                        rag_config = file_config["rag"]
                        config.update({
                            "backend": rag_config.get("backend", config["backend"]),
                            "local_storage_path": rag_config.get("local_storage_path", config["local_storage_path"]),
                        })
                        
                        for backend in ["snowflake", "qdrant", "chromadb", "faiss"]:
                            if backend in rag_config:
                                config[backend].update(rag_config[backend])
                        
                        if "embeddings" in rag_config:
                            config["embeddings"].update(rag_config["embeddings"])
            
            except Exception as e:
                logger.warning(f"Error loading config file {self.config_path}: {e}")
        
        return config
    
    def get_backend(self) -> str:
        """Get configured backend name"""
        return self.config["backend"]
    
    def get_local_storage_path(self) -> Path:
        """Get local storage path"""
        return Path(self.config["local_storage_path"])
    
    def get_backend_config(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific backend"""
        backend = backend or self.config["backend"]
        return self.config.get(backend, {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding provider configuration"""
        return self.config["embeddings"]
    
    def save_config(self, path: Optional[Path] = None) -> None:
        """Save configuration to file"""
        save_path = path or self.config_path or Path.cwd() / ".datamanagement_genai" / "config.toml"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to TOML format
        toml_content = f"""[rag]
backend = "{self.config['backend']}"
local_storage_path = "{self.config['local_storage_path']}"

[rag.embeddings]
provider = "{self.config['embeddings']['provider']}"
model = "{self.config['embeddings']['model']}"
dimension = {self.config['embeddings']['dimension']}

[rag.snowflake]
database = "{self.config['snowflake']['database']}"
schema = "{self.config['snowflake']['schema']}"
table = "{self.config['snowflake']['table']}"

[rag.qdrant]
collection_name = "{self.config['qdrant']['collection_name']}"

[rag.chromadb]
collection_name = "{self.config['chromadb']['collection_name']}"

[rag.faiss]
index_name = "{self.config['faiss']['index_name']}"
"""
        
        with open(save_path, "w") as f:
            f.write(toml_content)
        
        logger.info(f"Saved configuration to: {save_path}")
