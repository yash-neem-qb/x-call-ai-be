"""
API routes for knowledge base management.
"""

import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.knowledge_schemas import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate, KnowledgeBaseResponse,
    KnowledgeBaseListResponse, KnowledgeBaseSearchRequest, KnowledgeBaseSearchResponse,
    KnowledgeBaseUploadRequest, KnowledgeBaseBulkUploadRequest, KnowledgeBaseStatsResponse,
    FileUploadRequest, FileUploadResponse, BulkFileUploadRequest, BulkFileUploadResponse,
    DocumentPreviewResponse, SupportedFormatsResponse
)
from app.core.auth import require_write_permission, require_read_permission
from app.services.knowledge_service import knowledge_service
from app.services.file_upload_service import file_upload_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/organizations", tags=["knowledge-base"])


@router.post("/{organization_id}/knowledge-base", response_model=KnowledgeBaseResponse)
async def create_knowledge_document(
    organization_id: uuid.UUID,
    document: KnowledgeBaseCreate,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Create a new knowledge base document.
    
    Args:
        organization_id: Organization ID
        document: Document data
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Created knowledge base document
    """
    try:
        organization, membership = org_data
        
        # Prepare document data
        document_data = {
            "title": document.title,
            "content": document.content,
            "source": document.source,
            "assistant_id": document.assistant_id,
            "metadata": document.metadata,
            "tags": document.tags
        }
        
        # Create document
        db_document = await knowledge_service.create_document(
            db=db,
            organization_id=str(organization.id),
            document_data=document_data
        )
        
        logger.info(f"Created knowledge base document: {db_document.id}")
        return db_document
        
    except Exception as e:
        logger.error(f"Error creating knowledge base document: {e}")
        raise HTTPException(status_code=500, detail="Failed to create knowledge base document")


@router.get("/{organization_id}/knowledge-base", response_model=KnowledgeBaseListResponse)
async def list_knowledge_documents(
    organization_id: uuid.UUID,
    assistant_id: Optional[uuid.UUID] = Query(None, description="Filter by assistant ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of records"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    List knowledge base documents.
    
    Args:
        organization_id: Organization ID
        assistant_id: Optional assistant ID filter
        skip: Number of records to skip
        limit: Maximum number of records
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        List of knowledge base documents
    """
    try:
        organization, membership = org_data
        
        documents, total = await knowledge_service.list_documents(
            db=db,
            organization_id=str(organization.id),
            assistant_id=str(assistant_id) if assistant_id else None,
            skip=skip,
            limit=limit
        )
        
        
        return KnowledgeBaseListResponse(
            items=documents,
            total=total,
            page=(skip // limit) + 1,
            page_size=limit,
            total_pages=(total + limit - 1) // limit
        )
        
    except Exception as e:
        logger.error(f"Error listing knowledge base documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list knowledge base documents")


@router.get("/{organization_id}/knowledge-base/{document_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_document(
    organization_id: uuid.UUID,
    document_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get a knowledge base document by ID.
    
    Args:
        organization_id: Organization ID
        document_id: Document ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Knowledge base document
    """
    try:
        organization, membership = org_data
        
        document = await knowledge_service.get_document(
            db=db,
            document_id=str(document_id),
            organization_id=str(organization.id)
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Knowledge base document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base document: {e}")
        raise HTTPException(status_code=500, detail="Failed to get knowledge base document")


@router.put("/{organization_id}/knowledge-base/{document_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_document(
    organization_id: uuid.UUID,
    document_id: uuid.UUID,
    document_update: KnowledgeBaseUpdate,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Update a knowledge base document.
    
    Args:
        organization_id: Organization ID
        document_id: Document ID
        document_update: Update data
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Updated knowledge base document
    """
    try:
        organization, membership = org_data
        
        # Prepare update data
        update_data = {k: v for k, v in document_update.dict().items() if v is not None}
        
        document = await knowledge_service.update_document(
            db=db,
            document_id=str(document_id),
            organization_id=str(organization.id),
            update_data=update_data
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Knowledge base document not found")
        
        logger.info(f"Updated knowledge base document: {document_id}")
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating knowledge base document: {e}")
        raise HTTPException(status_code=500, detail="Failed to update knowledge base document")


@router.delete("/{organization_id}/knowledge-base/{document_id}")
async def delete_knowledge_document(
    organization_id: uuid.UUID,
    document_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Delete a knowledge base document.
    
    Args:
        organization_id: Organization ID
        document_id: Document ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Success message
    """
    try:
        organization, membership = org_data
        
        success = await knowledge_service.delete_document(
            db=db,
            document_id=str(document_id),
            organization_id=str(organization.id)
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Knowledge base document not found")
        
        logger.info(f"Deleted knowledge base document: {document_id}")
        return {"success": True, "message": f"Knowledge base document {document_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge base document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete knowledge base document")


@router.post("/{organization_id}/knowledge-base/search", response_model=KnowledgeBaseSearchResponse)
async def search_knowledge_base(
    organization_id: uuid.UUID,
    search_request: KnowledgeBaseSearchRequest,
    org_data: tuple = Depends(require_read_permission())
):
    """
    Search knowledge base documents using semantic search.
    
    Args:
        organization_id: Organization ID
        search_request: Search request data
        org_data: Organization and membership data
        
    Returns:
        Search results
    """
    try:
        organization, membership = org_data
        
        results = await knowledge_service.search_documents(
            organization_id=str(organization.id),
            query=search_request.query,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
            assistant_id=str(search_request.assistant_id) if search_request.assistant_id else None
        )
        
        return KnowledgeBaseSearchResponse(
            results=results,
            total=len(results),
            query=search_request.query,
            search_time_ms=results[0].get("search_time_ms", 0) if results else 0
        )
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Failed to search knowledge base")


@router.post("/{organization_id}/knowledge-base/upload", response_model=KnowledgeBaseResponse)
async def upload_knowledge_document(
    organization_id: uuid.UUID,
    upload_request: KnowledgeBaseUploadRequest,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Upload a knowledge base document with chunking.
    
    Args:
        organization_id: Organization ID
        upload_request: Upload request data
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Created knowledge base document
    """
    try:
        organization, membership = org_data
        
        # Prepare document data
        document_data = {
            "title": upload_request.title,
            "content": upload_request.content,
            "source": upload_request.source,
            "assistant_id": upload_request.assistant_id,
            "metadata": upload_request.metadata,
            "tags": upload_request.tags,
            "chunk_size": upload_request.chunk_size,
            "chunk_overlap": upload_request.chunk_overlap
        }
        
        # Create document
        db_document = await knowledge_service.create_document(
            db=db,
            organization_id=str(organization.id),
            document_data=document_data
        )
        
        logger.info(f"Uploaded knowledge base document: {db_document.id}")
        return db_document
        
    except Exception as e:
        logger.error(f"Error uploading knowledge base document: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload knowledge base document")


@router.post("/{organization_id}/knowledge-base/bulk-upload", response_model=List[KnowledgeBaseResponse])
async def bulk_upload_knowledge_documents(
    organization_id: uuid.UUID,
    bulk_request: KnowledgeBaseBulkUploadRequest,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Bulk upload multiple knowledge base documents.
    
    Args:
        organization_id: Organization ID
        bulk_request: Bulk upload request data
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        List of created knowledge base documents
    """
    try:
        organization, membership = org_data
        
        # Prepare documents data
        documents_data = []
        for doc in bulk_request.documents:
            documents_data.append({
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "assistant_id": doc.assistant_id,
                "metadata": doc.metadata,
                "tags": doc.tags,
                "chunk_size": bulk_request.chunk_size,
                "chunk_overlap": bulk_request.chunk_overlap
            })
        
        # Bulk create documents
        created_documents = await knowledge_service.bulk_upload_documents(
            db=db,
            organization_id=str(organization.id),
            documents=documents_data
        )
        
        logger.info(f"Bulk uploaded {len(created_documents)} knowledge base documents")
        return created_documents
        
    except Exception as e:
        logger.error(f"Error bulk uploading knowledge base documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to bulk upload knowledge base documents")


@router.get("/{organization_id}/knowledge-base/stats", response_model=KnowledgeBaseStatsResponse)
async def get_knowledge_base_stats(
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get knowledge base statistics.
    
    Args:
        organization_id: Organization ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Knowledge base statistics
    """
    try:
        organization, membership = org_data
        
        stats = await knowledge_service.get_knowledge_stats(
            db=db,
            organization_id=str(organization.id)
        )
        
        return KnowledgeBaseStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get knowledge base statistics")


# File Upload endpoints
@router.get("/{organization_id}/knowledge-base/supported-formats", response_model=SupportedFormatsResponse)
async def get_supported_formats(
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission())
):
    """
    Get supported file formats for upload.
    
    Args:
        organization_id: Organization ID
        org_data: Organization and membership data
        
    Returns:
        Supported formats information
    """
    try:
        organization, membership = org_data
        
        return SupportedFormatsResponse(
            supported_formats=file_upload_service.get_supported_formats(),
            max_file_size_mb=file_upload_service.get_max_file_size(),
            max_file_size_bytes=file_upload_service.get_max_file_size() * 1024 * 1024
        )
        
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported formats")


@router.post("/{organization_id}/knowledge-base/upload-file", response_model=FileUploadResponse)
async def upload_file(
    organization_id: uuid.UUID,
    file: UploadFile = File(...),
    assistant_id: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    tags: Optional[str] = Form(None),  # JSON string
    metadata: Optional[str] = Form(None),  # JSON string
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Upload a single file to the knowledge base.
    
    Args:
        organization_id: Organization ID
        file: Uploaded file
        assistant_id: Optional assistant ID
        chunk_size: Chunk size for processing
        chunk_overlap: Chunk overlap
        tags: Optional tags as JSON string
        metadata: Optional metadata as JSON string
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Upload result
    """
    try:
        organization, membership = org_data
        
        # Parse optional parameters
        import json
        parsed_tags = json.loads(tags) if tags else []
        parsed_metadata = json.loads(metadata) if metadata else {}
        
        # Convert assistant_id to UUID if provided
        assistant_uuid = uuid.UUID(assistant_id) if assistant_id else None
        
        # Upload and process file
        result = await file_upload_service.upload_and_process_document(
            file=file,
            organization_id=str(organization.id),
            assistant_id=str(assistant_uuid) if assistant_uuid else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tags=parsed_tags,
            metadata=parsed_metadata,
            db=db
        )
        
        # Get parsed content for response
        parsed_content = result["parsed_content"]
        
        # Create response object
        response = FileUploadResponse(
            success=True,
            knowledge_document_id=result.get("knowledge_document_id"),
            filename=file.filename,
            file_type=parsed_content.get("file_type", "unknown"),
            file_size=parsed_content.get("file_size", 0),
            title=parsed_content.get("title", file.filename),
            content_preview=parsed_content["content"][:500] + "..." if len(parsed_content["content"]) > 500 else parsed_content["content"],
            estimated_chunks=len(file_upload_service.embedding_service.chunk_text(
                parsed_content["content"], chunk_size=chunk_size, overlap=chunk_overlap
            )),
            message=result["message"]
        )
        
        # Log successful upload
        logger.info(f"File upload successful: {response.filename} -> {response.knowledge_document_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/{organization_id}/knowledge-base/upload-files", response_model=BulkFileUploadResponse)
async def upload_files(
    organization_id: uuid.UUID,
    files: List[UploadFile] = File(...),
    assistant_id: Optional[str] = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    tags: Optional[str] = Form(None),  # JSON string
    metadata: Optional[str] = Form(None),  # JSON string
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Upload multiple files to the knowledge base.
    
    Args:
        organization_id: Organization ID
        files: List of uploaded files
        assistant_id: Optional assistant ID
        chunk_size: Chunk size for processing
        chunk_overlap: Chunk overlap
        tags: Optional tags as JSON string
        metadata: Optional metadata as JSON string
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Bulk upload result
    """
    try:
        organization, membership = org_data
        
        # Parse optional parameters
        import json
        parsed_tags = json.loads(tags) if tags else []
        parsed_metadata = json.loads(metadata) if metadata else {}
        
        # Convert assistant_id to UUID if provided
        assistant_uuid = uuid.UUID(assistant_id) if assistant_id else None
        
        # Bulk upload and process files
        result = await file_upload_service.bulk_upload_documents(
            files=files,
            organization_id=str(organization.id),
            assistant_id=str(assistant_uuid) if assistant_uuid else None,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tags=parsed_tags,
            metadata=parsed_metadata,
            db=db
        )
        
        return BulkFileUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Error bulk uploading files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk upload files: {str(e)}")


@router.post("/{organization_id}/knowledge-base/preview-file", response_model=DocumentPreviewResponse)
async def preview_file(
    organization_id: uuid.UUID,
    file: UploadFile = File(...),
    org_data: tuple = Depends(require_read_permission())
):
    """
    Preview a file without uploading it to the knowledge base.
    
    Args:
        organization_id: Organization ID
        file: File to preview
        org_data: Organization and membership data
        
    Returns:
        File preview information
    """
    try:
        organization, membership = org_data
        
        # Preview file
        preview = await file_upload_service.preview_document(file)
        
        return DocumentPreviewResponse(**preview)
        
    except Exception as e:
        logger.error(f"Error previewing file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preview file: {str(e)}")
