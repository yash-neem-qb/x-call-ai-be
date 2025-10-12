"""
API routes for RAG (Retrieval-Augmented Generation) configuration management.
"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.rag_schemas import (
    RAGConfigSchema, RAGUsageStats, KnowledgeBaseSearchRequest, 
    KnowledgeBaseSearchResponse, KnowledgeBaseSearchResult
)
from app.core.auth import require_write_permission, require_read_permission
from app.services.knowledge_service import knowledge_service
from app.db.crud import get_assistant, update_assistant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/organizations", tags=["rag"])


@router.get("/{organization_id}/assistants/{assistant_id}/rag-config", response_model=RAGConfigSchema)
async def get_rag_config(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get RAG configuration for an assistant.
    
    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        RAG configuration
    """
    try:
        organization, membership = org_data
        
        # Get assistant
        assistant = get_assistant(db, assistant_id)
        if not assistant or assistant.organization_id != organization.id:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # Build RAG config from assistant data
        rag_config = RAGConfigSchema(
            enable_rag=assistant.rag_enabled,
            max_knowledge_results=assistant.rag_max_results,
            knowledge_score_threshold=float(assistant.rag_score_threshold),
            max_knowledge_context_length=assistant.rag_max_context_length,
            **assistant.rag_config or {}
        )
        
        return rag_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RAG config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RAG configuration")


@router.put("/{organization_id}/assistants/{assistant_id}/rag-config", response_model=RAGConfigSchema)
async def update_rag_config(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    rag_config: RAGConfigSchema,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Update RAG configuration for an assistant.
    
    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        rag_config: New RAG configuration
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Updated RAG configuration
    """
    try:
        organization, membership = org_data
        
        # Get assistant
        assistant = get_assistant(db, assistant_id)
        if not assistant or assistant.organization_id != organization.id:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # Update assistant with new RAG configuration
        update_data = {
            "rag_enabled": rag_config.enable_rag,
            "rag_max_results": rag_config.max_knowledge_results,
            "rag_score_threshold": str(rag_config.knowledge_score_threshold),
            "rag_max_context_length": rag_config.max_knowledge_context_length,
            "rag_config": {
                "include_knowledge_in_system_prompt": rag_config.include_knowledge_in_system_prompt,
                "knowledge_context_template": rag_config.knowledge_context_template,
                "fallback_to_general_llm": rag_config.fallback_to_general_llm,
                "log_knowledge_usage": rag_config.log_knowledge_usage,
                "search_timeout_seconds": rag_config.search_timeout_seconds,
                "cache_knowledge_results": rag_config.cache_knowledge_results,
                "cache_ttl_seconds": rag_config.cache_ttl_seconds
            }
        }
        
        updated_assistant = update_assistant(db, assistant_id, update_data)
        if not updated_assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        logger.info(f"Updated RAG config for assistant: {assistant_id}")
        return rag_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating RAG config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update RAG configuration")


@router.post("/{organization_id}/assistants/{assistant_id}/rag/search", response_model=KnowledgeBaseSearchResponse)
async def search_knowledge_base_for_assistant(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    search_request: KnowledgeBaseSearchRequest,
    org_data: tuple = Depends(require_read_permission())
):
    """
    Search knowledge base for a specific assistant.
    
    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        search_request: Search request data
        org_data: Organization and membership data
        
    Returns:
        Knowledge base search results
    """
    try:
        organization, membership = org_data
        
        # Search knowledge base with assistant filtering
        results = await knowledge_service.search_documents(
            organization_id=str(organization.id),
            query=search_request.query,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            assistant_id=str(assistant_id)
        )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(KnowledgeBaseSearchResult(
                id=result.get("id", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                source=result.get("source"),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                assistant_id=result.get("assistant_id")
            ))
        
        return KnowledgeBaseSearchResponse(
            results=search_results,
            total=len(search_results),
            query=search_request.query,
            search_time_ms=results[0].get("search_time_ms", 0) if results else 0,
            organization_id=str(organization.id),
            assistant_id=str(assistant_id)
        )
        
    except Exception as e:
        logger.error(f"Error searching knowledge base for assistant: {e}")
        raise HTTPException(status_code=500, detail="Failed to search knowledge base")


@router.get("/{organization_id}/assistants/{assistant_id}/rag/stats", response_model=RAGUsageStats)
async def get_rag_usage_stats(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get RAG usage statistics for an assistant.
    
    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        RAG usage statistics
    """
    try:
        organization, membership = org_data
        
        # Get assistant
        assistant = get_assistant(db, assistant_id)
        if not assistant or assistant.organization_id != organization.id:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # TODO: Implement actual RAG usage statistics collection
        # For now, return placeholder data
        stats = RAGUsageStats(
            total_queries=0,
            successful_rag_queries=0,
            fallback_queries=0,
            average_knowledge_results=0.0,
            average_processing_time=0.0,
            most_used_sources=[]
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting RAG usage stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RAG usage statistics")


@router.post("/{organization_id}/rag/test-search", response_model=KnowledgeBaseSearchResponse)
async def test_rag_search(
    organization_id: uuid.UUID,
    search_request: KnowledgeBaseSearchRequest,
    org_data: tuple = Depends(require_read_permission())
):
    """
    Test RAG search functionality for an organization.
    
    Args:
        organization_id: Organization ID
        search_request: Search request data
        org_data: Organization and membership data
        
    Returns:
        Knowledge base search results
    """
    try:
        organization, membership = org_data
        
        # Search knowledge base
        results = await knowledge_service.search_documents(
            organization_id=str(organization.id),
            query=search_request.query,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            assistant_id=str(search_request.assistant_id) if search_request.assistant_id else None
        )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(KnowledgeBaseSearchResult(
                id=result.get("id", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                source=result.get("source"),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                assistant_id=result.get("assistant_id")
            ))
        
        return KnowledgeBaseSearchResponse(
            results=search_results,
            total=len(search_results),
            query=search_request.query,
            search_time_ms=results[0].get("search_time_ms", 0) if results else 0,
            organization_id=str(organization.id),
            assistant_id=search_request.assistant_id
        )
        
    except Exception as e:
        logger.error(f"Error testing RAG search: {e}")
        raise HTTPException(status_code=500, detail="Failed to test RAG search")
