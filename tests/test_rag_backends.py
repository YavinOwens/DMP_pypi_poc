"""
Tests for RAG backend functionality
"""

import pytest

from datamanagement_genai.rag.backends.factory import create_vector_store
from datamanagement_genai.rag.config import RAGConfig


class TestRAGBackends:
    """Test RAG backend factory and configuration"""

    def test_rag_config_creation(self):
        """Test that RAGConfig can be created"""
        config = RAGConfig()
        assert config is not None
        assert hasattr(config, 'config')

    def test_factory_function_exists(self):
        """Test that factory function exists"""
        assert callable(create_vector_store)

    def test_factory_without_session(self):
        """Test that factory handles missing session gracefully"""
        config = RAGConfig()
        # Should raise ValueError for snowflake backend without session
        with pytest.raises(ValueError, match="Snowflake session required"):
            create_vector_store(backend="snowflake", config=config)

    def test_factory_unknown_backend(self):
        """Test that factory raises error for unknown backend"""
        config = RAGConfig()
        with pytest.raises(ValueError, match="Unknown backend"):
            create_vector_store(backend="unknown_backend", config=config)
