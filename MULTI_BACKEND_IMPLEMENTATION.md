# Multi-Backend Implementation Guide

## Overview

The package now supports multiple vector database backends with local storage capabilities. This is **best practice** for several reasons:

### ✅ Why This Is Best Practice

1. **Separation of Concerns**: Abstract interfaces allow swapping implementations
2. **Testability**: Local backends enable testing without cloud dependencies
3. **Flexibility**: Users choose the best backend for their use case
4. **Cost Efficiency**: Local backends for development/testing
5. **Scalability**: Cloud backends for production workloads
6. **SOLID Principles**: Follows Open/Closed and Dependency Inversion principles

### 📁 Local Storage Structure

When using local backends, data is stored in:

```
{project_root}/
└── .datamanagement_genai/          # Created automatically
    ├── config.db                   # SQLite for metadata/config
    └── vector_stores/              # Vector store data
        ├── qdrant/                 # Qdrant data
        ├── chromadb/                # ChromaDB data
        └── faiss/                   # FAISS indices + metadata DB
```

**This is NOT overhead** - it's a feature that enables:
- Isolated testing environments
- Offline development
- Easy cleanup (just delete `.datamanagement_genai/`)
- Version control friendly (can be gitignored)

## Configuration

### Option 1: Config File (Recommended)

Create `config.toml` or `.datamanagement_genai/config.toml`:

```toml
[rag]
backend = "qdrant"  # Options: snowflake, qdrant, chromadb, faiss
local_storage_path = ".datamanagement_genai"

[rag.embeddings]
provider = "snowflake"  # Options: snowflake, openai, huggingface, local
model = "snowflake-arctic-embed-l-v2.0"
dimension = 768

[rag.qdrant]
collection_name = "knowledge_base"

[rag.chromadb]
collection_name = "knowledge_base"

[rag.faiss]
index_name = "knowledge_base"

[rag.snowflake]
database = "RAG_KNOWLEDGE_BASE"
schema = "PUBLIC"
table = "VECTOR_STORE"
```

### Option 2: Environment Variables

```bash
export RAG_BACKEND=qdrant
export RAG_LOCAL_STORAGE=.datamanagement_genai
export RAG_EMBEDDING_PROVIDER=snowflake
export RAG_EMBEDDING_MODEL=snowflake-arctic-embed-l-v2.0
```

## Usage Examples

### Using Qdrant (Local)

```python
from datamanagement_genai.rag.backends.factory import create_vector_store, create_embedding_provider
from datamanagement_genai.rag.config import RAGConfig

# Configure for Qdrant
config = RAGConfig()
config.config["backend"] = "qdrant"
config.config["embeddings"]["dimension"] = 768

# Create vector store (local, no Snowflake needed)
vector_store = create_vector_store(backend="qdrant", config=config)
vector_store.initialize()

# For embeddings, still need Snowflake session (or use local provider)
# session = get_snowflake_session()  # Optional if using Snowflake embeddings
# embedding_provider = create_embedding_provider(provider="snowflake", session=session, config=config)

# Or use local embeddings (when implemented)
# embedding_provider = create_embedding_provider(provider="local", config=config)

# Add vectors
ids = vector_store.add_vectors(
    vectors=[[0.1, 0.2, ...], ...],
    texts=["Document 1", "Document 2"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
)

# Search
results = vector_store.search(
    query_vector=[0.1, 0.2, ...],
    top_k=5
)
```

### Using ChromaDB (Local)

```python
# Same pattern, just change backend
vector_store = create_vector_store(backend="chromadb", config=config)
vector_store.initialize()
```

### Using FAISS (Local)

```python
# Same pattern
vector_store = create_vector_store(backend="faiss", config=config)
vector_store.initialize()
```

### Using Snowflake (Existing)

```python
from datamanagement_genai import get_snowflake_session

session = get_snowflake_session()
vector_store = create_vector_store(backend="snowflake", session=session)
vector_store.initialize()
```

## Installation

### Basic Installation

```bash
pip install datamanagement-genai
```

### With Specific Backends

```bash
# Qdrant
pip install datamanagement-genai[qdrant]

# ChromaDB
pip install datamanagement-genai[chromadb]

# FAISS
pip install datamanagement-genai[faiss]

# All local backends
pip install datamanagement-genai[qdrant,chromadb,faiss]

# With local embeddings (when implemented)
pip install datamanagement-genai[local-embeddings]
```

## SQLite Database

The `.datamanagement_genai/config.db` SQLite database stores:

1. **Vector Store Configuration**: Backend settings, dimensions, etc.
2. **Document Metadata**: Mapping between documents and vector IDs
3. **Cross-Backend Tracking**: Track documents across different backends

This is **NOT overhead** - it provides:
- **Metadata Management**: Track document sources, chunk indices
- **Cross-Backend Queries**: Find documents regardless of backend
- **Configuration Persistence**: Remember settings between sessions
- **Migration Support**: Move data between backends

## Best Practices

### ✅ DO

- Use local backends for development/testing
- Use Snowflake for production (if you have Snowflake)
- Use Qdrant/ChromaDB for production (if you need local/self-hosted)
- Store config in `.datamanagement_genai/config.toml`
- Add `.datamanagement_genai/` to `.gitignore`

### ❌ DON'T

- Commit `.datamanagement_genai/` to git (contains data)
- Mix backends in the same collection (use different collection names)
- Use production Snowflake credentials for local testing

## Migration Between Backends

To migrate data between backends:

```python
# Export from source
source_store = create_vector_store(backend="qdrant")
results = source_store.search(query_vector=[0]*768, top_k=10000)  # Get all

# Import to destination
dest_store = create_vector_store(backend="chromadb")
dest_store.add_vectors(
    vectors=[r['embedding'] for r in results],
    texts=[r['text'] for r in results],
    metadatas=[r['metadata'] for r in results]
)
```

## Performance Considerations

- **FAISS**: Fastest for large-scale similarity search
- **Qdrant**: Good balance of features and performance
- **ChromaDB**: Easy to use, good for small-medium datasets
- **Snowflake**: Best for cloud-native, integrated workflows

## Troubleshooting

### Backend Not Available

If you get `ImportError`, install the required package:

```bash
pip install qdrant-client      # For Qdrant
pip install chromadb           # For ChromaDB
pip install faiss-cpu          # For FAISS (or faiss-gpu)
```

### Local Storage Issues

If local storage isn't working:

1. Check permissions on `.datamanagement_genai/` directory
2. Ensure disk space is available
3. Check SQLite database isn't locked (close other connections)

### Dimension Mismatches

Ensure embedding dimension matches vector store dimension:

```python
config = RAGConfig()
config.config["embeddings"]["dimension"] = 768  # Must match model
vector_store = create_vector_store(config=config)  # Uses config dimension
```

## Next Steps

1. **Update RAG System**: Refactor `RAGSystem` to use the factory
2. **Add Local Embeddings**: Implement HuggingFace/sentence-transformers provider
3. **Add Migration Tools**: Utilities to move data between backends
4. **Add Tests**: Comprehensive tests for all backends
5. **Update Documentation**: Add examples for each backend
