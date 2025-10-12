"""
File upload service for handling document uploads and processing.
"""

import logging
import uuid
import os
from typing import Dict, Any, List, Optional
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.services.document_parser_service import document_parser_service
from app.services.knowledge_service import knowledge_service
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class FileUploadService:
    """Service for handling file uploads and document processing."""
    
    def __init__(self):
        self.document_parser = document_parser_service
        self.knowledge_service = knowledge_service
        self.embedding_service = embedding_service
        
        # File size limits (in MB)
        self.max_file_size = 50
        self.supported_formats = self.document_parser.get_supported_formats()
    
    async def upload_and_process_document(
        self,
        file: UploadFile,
        organization_id: str,
        assistant_id: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Upload and process a document file.
        
        Args:
            file: Uploaded file
            organization_id: Organization ID
            assistant_id: Optional assistant ID
            chunk_size: Chunk size for processing
            chunk_overlap: Chunk overlap
            tags: Optional tags
            metadata: Optional metadata
            db: Database session
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting file upload process for: {file.filename}")
            
            # Validate file
            await self._validate_file(file)
            
            # Read file content
            file_content = await file.read()
            
            # Parse document
            parsed_content = await self.document_parser.parse_document(
                file_content, file.filename
            )
            
            # Prepare document data for knowledge base
            document_data = {
                "title": parsed_content.get("title", file.filename),
                "content": parsed_content["content"],
                "source": f"uploaded_file:{file.filename}",
                "assistant_id": assistant_id,
                "metadata": {
                    **(metadata or {}),
                    "file_metadata": parsed_content.get("metadata", {}),
                    "file_structure": parsed_content.get("structure", {}),
                    "original_filename": file.filename,
                    "file_type": parsed_content.get("file_type"),
                    "file_size": parsed_content.get("file_size"),
                    "parsing_method": parsed_content.get("parsing_method")
                },
                "tags": (tags or []) + [parsed_content.get("file_type", "unknown")],
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            
            # Create knowledge base document
            if db:
                knowledge_document = await self.knowledge_service.create_document(
                    db=db,
                    organization_id=organization_id,
                    document_data=document_data
                )
                
                return {
                    "success": True,
                    "knowledge_document_id": str(knowledge_document.id),
                    "parsed_content": parsed_content,
                    "message": f"Successfully processed and uploaded {file.filename}"
                }
            else:
                # Return parsed content without saving to database
                return {
                    "success": True,
                    "parsed_content": parsed_content,
                    "message": f"Successfully parsed {file.filename}"
                }
                
        except Exception as e:
            logger.error(f"Error uploading and processing file {file.filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process file {file.filename}: {str(e)}"
            )
    
    async def bulk_upload_documents(
        self,
        files: List[UploadFile],
        organization_id: str,
        assistant_id: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Bulk upload and process multiple documents.
        
        Args:
            files: List of uploaded files
            organization_id: Organization ID
            assistant_id: Optional assistant ID
            chunk_size: Chunk size for processing
            chunk_overlap: Chunk overlap
            tags: Optional tags
            metadata: Optional metadata
            db: Database session
            
        Returns:
            Dictionary with bulk processing results
        """
        try:
            results = []
            successful_uploads = 0
            failed_uploads = 0
            
            for file in files:
                try:
                    result = await self.upload_and_process_document(
                        file=file,
                        organization_id=organization_id,
                        assistant_id=assistant_id,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        tags=tags,
                        metadata=metadata,
                        db=db
                    )
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "result": result
                    })
                    successful_uploads += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file.filename}: {e}")
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": str(e)
                    })
                    failed_uploads += 1
            
            return {
                "success": True,
                "total_files": len(files),
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "results": results,
                "message": f"Processed {successful_uploads}/{len(files)} files successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in bulk upload: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Bulk upload failed: {str(e)}"
            )
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        # Check file size
        if not self.document_parser.validate_file_size(
            await file.read(), self.max_file_size
        ):
            await file.seek(0)  # Reset file pointer
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {self.max_file_size}MB"
            )
        
        # Reset file pointer after size check
        await file.seek(0)
        
        # Check file extension
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in self.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}"
            )
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return self.supported_formats
    
    def get_max_file_size(self) -> int:
        """Get maximum file size in MB."""
        return self.max_file_size
    
    async def preview_document(self, file: UploadFile) -> Dict[str, Any]:
        """
        Preview document content without saving to database.
        
        Args:
            file: Uploaded file
            
        Returns:
            Dictionary with preview information
        """
        try:
            # Validate file
            await self._validate_file(file)
            
            # Read and parse file
            file_content = await file.read()
            parsed_content = await self.document_parser.parse_document(
                file_content, file.filename
            )
            
            # Generate preview (first 1000 characters)
            content_preview = parsed_content["content"][:1000]
            if len(parsed_content["content"]) > 1000:
                content_preview += "..."
            
            return {
                "success": True,
                "filename": file.filename,
                "file_type": parsed_content.get("file_type"),
                "file_size": parsed_content.get("file_size"),
                "title": parsed_content.get("title"),
                "content_preview": content_preview,
                "metadata": parsed_content.get("metadata", {}),
                "structure": parsed_content.get("structure", {}),
                "estimated_chunks": len(self.embedding_service.chunk_text(
                    parsed_content["content"], chunk_size=1000, overlap=200
                ))
            }
            
        except Exception as e:
            logger.error(f"Error previewing document {file.filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to preview document: {str(e)}"
            )


# Global instance
file_upload_service = FileUploadService()
