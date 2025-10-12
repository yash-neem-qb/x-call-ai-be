"""
Pydantic schemas for knowledge base operations.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class KnowledgeBaseBase(BaseModel):
    """Base schema for knowledge base documents."""
    title: str = Field(..., max_length=500, description="Document title")
    content: str = Field(..., description="Document content")
    source: Optional[str] = Field(None, max_length=255, description="Source URL or file path")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema for creating a new knowledge base document."""
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant ID (optional)")


class KnowledgeBaseUpdate(BaseModel):
    """Schema for updating a knowledge base document."""
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    source: Optional[str] = Field(None, max_length=255)
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None


class KnowledgeBaseResponse(KnowledgeBaseBase):
    """Schema for knowledge base document response."""
    id: uuid.UUID
    organization_id: uuid.UUID
    assistant_id: Optional[uuid.UUID]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class KnowledgeBaseSearchRequest(BaseModel):
    """Schema for knowledge base search request."""
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of results")
    score_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Filter by assistant ID")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")


class KnowledgeBaseSearchResult(BaseModel):
    """Schema for knowledge base search result."""
    id: str
    score: float
    content: str
    title: str
    source: Optional[str]
    metadata: Dict[str, Any]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class KnowledgeBaseSearchResponse(BaseModel):
    """Schema for knowledge base search response."""
    results: List[KnowledgeBaseSearchResult]
    total: int
    query: str
    search_time_ms: float


class KnowledgeBaseListResponse(BaseModel):
    """Schema for knowledge base list response."""
    items: List[KnowledgeBaseResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class KnowledgeBaseUploadRequest(BaseModel):
    """Schema for knowledge base upload request."""
    title: str = Field(..., max_length=500, description="Document title")
    content: str = Field(..., description="Document content")
    source: Optional[str] = Field(None, max_length=255, description="Source URL or file path")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant ID (optional)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    chunk_size: Optional[int] = Field(1000, ge=100, le=2000, description="Chunk size for processing")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=500, description="Chunk overlap")


class KnowledgeBaseBulkUploadRequest(BaseModel):
    """Schema for bulk knowledge base upload request."""
    documents: List[KnowledgeBaseUploadRequest]
    chunk_size: Optional[int] = Field(1000, ge=100, le=2000, description="Chunk size for processing")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=500, description="Chunk overlap")


class KnowledgeBaseStatsResponse(BaseModel):
    """Schema for knowledge base statistics response."""
    total_documents: int
    total_chunks: int
    organization_id: uuid.UUID
    assistant_stats: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class FileUploadRequest(BaseModel):
    """Schema for file upload request."""
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant ID (optional)")
    chunk_size: Optional[int] = Field(1000, ge=100, le=2000, description="Chunk size for processing")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=500, description="Chunk overlap")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    success: bool
    knowledge_document_id: Optional[str] = None
    filename: str
    file_type: str
    file_size: int
    title: str
    content_preview: str
    estimated_chunks: int
    message: str


class BulkFileUploadRequest(BaseModel):
    """Schema for bulk file upload request."""
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant ID (optional)")
    chunk_size: Optional[int] = Field(1000, ge=100, le=2000, description="Chunk size for processing")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=500, description="Chunk overlap")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class BulkFileUploadResponse(BaseModel):
    """Schema for bulk file upload response."""
    success: bool
    total_files: int
    successful_uploads: int
    failed_uploads: int
    results: List[Dict[str, Any]]
    message: str


class DocumentPreviewResponse(BaseModel):
    """Schema for document preview response."""
    success: bool
    filename: str
    file_type: str
    file_size: int
    title: str
    content_preview: str
    metadata: Dict[str, Any]
    structure: Dict[str, Any]
    estimated_chunks: int


class SupportedFormatsResponse(BaseModel):
    """Schema for supported formats response."""
    supported_formats: List[str]
    max_file_size_mb: int
    max_file_size_bytes: int
