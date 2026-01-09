"""
Snowflake vector store and embedding provider implementation

This is a refactored version of the existing Snowflake implementation
to fit the new abstraction layer.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base import VectorStore, EmbeddingProvider

logger = logging.getLogger(__name__)

try:
    from snowflake.snowpark import Session
    from snowflake.snowpark.functions import col, lit
    SNOWPARK_AVAILABLE = True
except ImportError:
    SNOWPARK_AVAILABLE = False
    Session = None


class SnowflakeEmbeddingProvider(EmbeddingProvider):
    """Snowflake embedding provider using AI_EMBED function"""
    
    def __init__(self, session: Any, model: str = "snowflake-arctic-embed-l-v2.0"):
        """
        Initialize Snowflake embedding provider
        
        Args:
            session: Snowflake session
            model: Embedding model name
        """
        if not SNOWPARK_AVAILABLE:
            raise ImportError("snowflake-snowpark-python required for Snowflake embeddings")
        
        self.session = session
        self._model = model
        self._dimension = self._get_dimension()
    
    def _get_dimension(self) -> int:
        """Get embedding dimension for the model"""
        dimension_map = {
            "e5-base-v2": 768,
            "snowflake-arctic-embed-l-v2.0": 1024,
            "snowflake-arctic-embed-l-v2.0-8k": 1024,
            "nv-embed-qa-4": 1024,
            "multilingual-e5-large": 1024,
        }
        return dimension_map.get(self._model, 1024)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Escape single quotes for SQL
            text_escaped = text.replace("'", "''").replace("\\", "\\\\")
            
            query = f"SELECT AI_EMBED('{self._model}', '{text_escaped}') AS embedding"
            result = self.session.sql(query).collect()
            
            if result and len(result) > 0:
                row = result[0]
                embedding = row[0] if isinstance(row, (list, tuple)) else row.get("EMBEDDING") or row.get("embedding")
                
                if embedding is None:
                    logger.warning("Could not extract embedding from result")
                    return None
                
                # Convert to list
                if isinstance(embedding, list):
                    embedding_list = embedding
                elif hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                # Verify dimension
                if len(embedding_list) != self._dimension:
                    logger.warning(f"Dimension mismatch: got {len(embedding_list)}, expected {self._dimension}")
                    if len(embedding_list) > self._dimension:
                        embedding_list = embedding_list[:self._dimension]
                    else:
                        embedding_list = list(embedding_list) + [0.0] * (self._dimension - len(embedding_list))
                
                return embedding_list
            
            logger.warning("No embedding returned from Snowflake")
            return None
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([0.0] * self._dimension)  # Fallback
        return embeddings


class SnowflakeVectorStore(VectorStore):
    """Snowflake vector store implementation"""
    
    def __init__(
        self,
        session: Any,
        table_name: str = "VECTOR_STORE",
        schema: str = "PUBLIC",
        database: Optional[str] = None,
        dimension: int = 1024
    ):
        """
        Initialize Snowflake vector store
        
        Args:
            session: Snowflake session
            table_name: Table name for vector store
            schema: Schema name
            database: Database name (optional)
            dimension: Vector dimension
        """
        if not SNOWPARK_AVAILABLE:
            raise ImportError("snowflake-snowpark-python required for Snowflake vector store")
        
        self.session = session
        self.table_name = table_name
        self.schema = schema
        self.database = database
        self.dimension = dimension
        self._table_path = self._get_table_path()
    
    def _get_table_path(self) -> str:
        """Get full table path"""
        if self.database:
            return f"{self.database}.{self.schema}.{self.table_name}"
        return f"{self.schema}.{self.table_name}"
    
    def initialize(self) -> None:
        """Create vector store table if it doesn't exist"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_path} (
            id VARCHAR(255) PRIMARY KEY,
            text TEXT,
            embedding VECTOR(FLOAT, {self.dimension}),
            metadata VARIANT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        try:
            self.session.sql(create_table_sql).collect()
            logger.info(f"✓ Vector store table ready: {self._table_path}")
        except Exception as e:
            logger.error(f"Error creating vector store table: {e}")
            raise
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to Snowflake"""
        if ids is None:
            ids = [f"chunk_{i}_{datetime.now().timestamp()}" for i in range(len(vectors))]
        
        if not (len(vectors) == len(texts) == len(metadatas) == len(ids)):
            raise ValueError("All input lists must have the same length")
        
        added_ids = []
        for vector, text, metadata, chunk_id in zip(vectors, texts, metadatas, ids):
            try:
                # Ensure correct dimension
                if len(vector) != self.dimension:
                    if len(vector) > self.dimension:
                        vector = vector[:self.dimension]
                    else:
                        vector = list(vector) + [0.0] * (self.dimension - len(vector))
                
                # Escape for SQL
                text_escaped = text.replace("'", "''").replace("\\", "\\\\")
                metadata_json = json.dumps(metadata).replace("'", "''")
                embedding_json = json.dumps([float(x) for x in vector])
                
                insert_sql = f"""
                INSERT INTO {self._table_path} (id, text, embedding, metadata)
                SELECT 
                    '{chunk_id}',
                    '{text_escaped}',
                    PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, {self.dimension}),
                    PARSE_JSON('{metadata_json}')
                """
                
                self.session.sql(insert_sql).collect()
                added_ids.append(chunk_id)
                
            except Exception as e:
                logger.error(f"Error adding vector {chunk_id}: {e}")
        
        return added_ids
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Ensure correct dimension
        if len(query_vector) != self.dimension:
            if len(query_vector) > self.dimension:
                query_vector = query_vector[:self.dimension]
            else:
                query_vector = list(query_vector) + [0.0] * (self.dimension - len(query_vector))
        
        query_vector_json = json.dumps([float(x) for x in query_vector])
        
        # Build query with optional filter
        where_clause = ""
        if filter:
            # Simple filter support (can be extended)
            conditions = []
            for key, value in filter.items():
                conditions.append(f"metadata:{key} = '{value}'")
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
        
        search_sql = f"""
        SELECT 
            id,
            text,
            metadata,
            VECTOR_L2_DISTANCE(embedding, PARSE_JSON('{query_vector_json}')::VECTOR(FLOAT, {self.dimension})) AS distance
        FROM {self._table_path}
        {where_clause}
        ORDER BY distance ASC
        LIMIT {top_k}
        """
        
        try:
            results = self.session.sql(search_sql).collect()
            
            formatted_results = []
            for row in results:
                result = {
                    'id': row[0] if isinstance(row, (list, tuple)) else row.get('ID') or row.get('id'),
                    'text': row[1] if isinstance(row, (list, tuple)) else row.get('TEXT') or row.get('text'),
                    'metadata': json.loads(row[2]) if isinstance(row[2], str) else row[2] if isinstance(row, (list, tuple)) else row.get('METADATA') or row.get('metadata'),
                    'score': float(row[3]) if isinstance(row, (list, tuple)) else float(row.get('DISTANCE') or row.get('distance', 0.0)),
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        if not ids:
            return True
        
        ids_str = "', '".join(id.replace("'", "''") for id in ids)
        delete_sql = f"DELETE FROM {self._table_path} WHERE id IN ('{ids_str}')"
        
        try:
            self.session.sql(delete_sql).collect()
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of vectors"""
        count_sql = f"SELECT COUNT(*) FROM {self._table_path}"
        try:
            result = self.session.sql(count_sql).collect()
            if result:
                return int(result[0][0])
            return 0
        except Exception as e:
            logger.error(f"Error counting vectors: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            self.session.sql(f"TRUNCATE TABLE {self._table_path}").collect()
            return True
        except Exception as e:
            logger.error(f"Error clearing vectors: {e}")
            return False
