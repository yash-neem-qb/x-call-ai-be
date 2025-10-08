"""
RAG (Retrieval-Augmented Generation) configuration schemas.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class RAGConfigSchema(BaseModel):
    """RAG configuration schema for assistant settings."""
    
    # Knowledge base search settings
    enable_rag: bool = Field(default=True, description="Enable RAG for this assistant")
    max_knowledge_results: int = Field(default=3, ge=1, le=10, description="Maximum number of knowledge base results to retrieve")
    knowledge_score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score for knowledge base results")
    max_knowledge_context_length: int = Field(default=2000, ge=500, le=5000, description="Maximum length of knowledge context to include")
    
    # System prompt enhancement
    include_knowledge_in_system_prompt: bool = Field(default=True, description="Include knowledge context in system prompt")
    knowledge_context_template: Optional[str] = Field(default=None, description="Custom template for knowledge context injection")
    
    # Fallback behavior
    fallback_to_general_llm: bool = Field(default=True, description="Fallback to general LLM if RAG fails")
    log_knowledge_usage: bool = Field(default=True, description="Log knowledge base usage for analytics")
    
    # Advanced settings
    search_timeout_seconds: int = Field(default=5, ge=1, le=30, description="Timeout for knowledge base search")
    cache_knowledge_results: bool = Field(default=False, description="Cache knowledge base results for similar queries")
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds")


class RAGUsageStats(BaseModel):
    """RAG usage statistics for analytics."""
    
    total_queries: int = Field(default=0, description="Total number of queries processed")
    successful_rag_queries: int = Field(default=0, description="Number of queries that used RAG successfully")
    fallback_queries: int = Field(default=0, description="Number of queries that fell back to general LLM")
    average_knowledge_results: float = Field(default=0.0, description="Average number of knowledge results per query")
    average_processing_time: float = Field(default=0.0, description="Average processing time in seconds")
    most_used_sources: List[str] = Field(default_factory=list, description="Most frequently used knowledge sources")


class KnowledgeBaseSearchRequest(BaseModel):
    """Request schema for knowledge base search."""
    
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    assistant_id: Optional[str] = Field(default=None, description="Assistant ID for filtering")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    include_metadata: bool = Field(default=True, description="Include document metadata in results")


class KnowledgeBaseSearchResult(BaseModel):
    """Knowledge base search result."""
    
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source: Optional[str] = Field(default=None, description="Document source")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: dict = Field(default_factory=dict, description="Document metadata")
    assistant_id: Optional[str] = Field(default=None, description="Associated assistant ID")


class KnowledgeBaseSearchResponse(BaseModel):
    """Response schema for knowledge base search."""
    
    results: List[KnowledgeBaseSearchResult] = Field(default_factory=list, description="Search results")
    total: int = Field(default=0, description="Total number of results found")
    query: str = Field(..., description="Original search query")
    search_time_ms: float = Field(default=0.0, description="Search time in milliseconds")
    organization_id: str = Field(..., description="Organization ID")
    assistant_id: Optional[str] = Field(default=None, description="Assistant ID used for filtering")
