"""
Vector database configuration for knowledge base.
"""

from app.config.settings import settings


class VectorConfig:
    """Vector database configuration using main settings."""
    
    # Qdrant Configuration
    qdrant_host: str = settings.vector_qdrant_host
    qdrant_port: int = settings.vector_qdrant_port
    qdrant_api_key: str = settings.vector_qdrant_api_key
    qdrant_url: str = settings.vector_qdrant_url
    
    # Collection Configuration
    knowledge_collection_name: str = settings.vector_knowledge_collection_name
    vector_size: int = settings.vector_size
    distance_metric: str = settings.vector_distance_metric
    
    # Embedding Configuration
    embedding_provider: str = settings.vector_embedding_provider
    openai_api_key: str = settings.openai_api_key
    openai_embedding_model: str = settings.vector_openai_embedding_model
    local_embedding_model: str = settings.vector_local_embedding_model
    
    # Chunking Configuration
    chunk_size: int = settings.vector_chunk_size
    chunk_overlap: int = settings.vector_chunk_overlap
    max_tokens_per_chunk: int = settings.vector_max_tokens_per_chunk
    
    # Search Configuration
    default_search_limit: int = settings.vector_default_search_limit
    similarity_threshold: float = settings.vector_similarity_threshold


# Global instance
vector_config = VectorConfig()
