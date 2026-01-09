# Multi-Backend Architecture Design

## Overview

This document outlines the design for supporting multiple vector database backends (Snowflake, Qdrant, ChromaDB, FAISS) with local storage support.

## Design Principles

1. **Abstraction Layer**: Abstract interfaces for vector stores and embedding providers
2. **Strategy Pattern**: Pluggable backends that can be swapped via configuration
3. **Local-First**: Support local storage for testing and development
4. **Backward Compatible**: Existing Snowflake-only code continues to work
5. **Configuration-Driven**: Choose backend via config file or environment variables

## Architecture

```
RAGSystem
    ├── VectorStore (Abstract Interface)
    │   ├── SnowflakeVectorStore
    │   ├── QdrantVectorStore
    │   ├── ChromaDBVectorStore
    │   └── FAISSVectorStore
    │
    └── EmbeddingProvider (Abstract Interface)
        ├── SnowflakeEmbeddingProvider (uses AI_EMBED)
        ├── OpenAIEmbeddingProvider
        ├── HuggingFaceEmbeddingProvider
        └── LocalEmbeddingProvider (uses sentence-transformers)
```

## Local Storage Structure

```
{project_root}/
├── .datamanagement_genai/          # Local storage directory
│   ├── config.db                   # SQLite config/metadata database
│   ├── vector_stores/              # Vector store data
│   │   ├── qdrant/                 # Qdrant data
│   │   ├── chromadb/                # ChromaDB data
│   │   └── faiss/                   # FAISS indices
│   └── embeddings/                  # Cached embeddings (optional)
```

## Benefits

✅ **Best Practice**: Follows SOLID principles (Open/Closed, Dependency Inversion)
✅ **Testability**: Easy to mock and test with local backends
✅ **Flexibility**: Users can choose the best backend for their use case
✅ **Cost Efficiency**: Local backends for development/testing
✅ **Scalability**: Can use cloud backends (Snowflake, Qdrant Cloud) for production

## Implementation Plan

1. Create abstract base classes
2. Implement Snowflake backend (refactor existing code)
3. Implement local backends (Qdrant, ChromaDB, FAISS)
4. Add configuration system
5. Add local storage management
6. Update RAG system to use abstraction
7. Add migration utilities

## Configuration

```toml
[rag]
backend = "snowflake"  # Options: snowflake, qdrant, chromadb, faiss
local_storage_path = ".datamanagement_genai"

[rag.snowflake]
database = "RAG_KNOWLEDGE_BASE"
schema = "PUBLIC"
table = "VECTOR_STORE"

[rag.qdrant]
path = ".datamanagement_genai/vector_stores/qdrant"
collection_name = "knowledge_base"

[rag.chromadb]
path = ".datamanagement_genai/vector_stores/chromadb"
collection_name = "knowledge_base"

[rag.faiss]
index_path = ".datamanagement_genai/vector_stores/faiss/knowledge_base.index"
metadata_path = ".datamanagement_genai/vector_stores/faiss/knowledge_base.db"

[rag.embeddings]
provider = "snowflake"  # Options: snowflake, openai, huggingface, local
model = "snowflake-arctic-embed-l-v2.0"
dimension = 768
```

## SQLite Database Schema

```sql
CREATE TABLE vector_store_config (
    id INTEGER PRIMARY KEY,
    backend TEXT NOT NULL,
    collection_name TEXT,
    dimension INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_metadata (
    id INTEGER PRIMARY KEY,
    document_id TEXT UNIQUE NOT NULL,
    source_path TEXT,
    chunk_index INTEGER,
    backend TEXT,
    vector_id TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
