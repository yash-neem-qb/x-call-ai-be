"""
Knowledge base service for managing documents and RAG operations.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.db.models import KnowledgeBase, Organization, Assistant
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service

logger = logging.getLogger(__name__)


class KnowledgeService:
    """Service for knowledge base operations."""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.vector_service = vector_service
    
    async def create_document(self, db: Session, organization_id: str, 
                            document_data: Dict[str, Any]) -> KnowledgeBase:
        """
        Create a new knowledge base document.
        
        Args:
            db: Database session
            organization_id: Organization ID
            document_data: Document data
            
        Returns:
            Created knowledge base document
        """
        try:
            # Create database record
            db_document = KnowledgeBase(
                organization_id=organization_id,
                assistant_id=document_data.get("assistant_id"),
                title=document_data["title"],
                content=document_data["content"],
                source=document_data.get("source"),
                document_metadata=document_data.get("metadata", {}),
                tags=document_data.get("tags", [])
            )
            
            db.add(db_document)
            db.flush()  # Get the ID
            
            # Process and store in vector database
            await self._process_and_store_document(
                db_document, 
                document_data.get("chunk_size", 1000),
                document_data.get("chunk_overlap", 200)
            )
            
            db.commit()
            db.refresh(db_document)
            
            logger.info(f"Created knowledge base document: {db_document.id}")
            return db_document
            
        except Exception as e:
            logger.error(f"Error creating knowledge base document: {e}")
            db.rollback()
            raise
    
    async def update_document(self, db: Session, document_id: str, 
                            organization_id: str, update_data: Dict[str, Any]) -> Optional[KnowledgeBase]:
        """
        Update a knowledge base document.
        
        Args:
            db: Database session
            document_id: Document ID
            organization_id: Organization ID
            update_data: Update data
            
        Returns:
            Updated document or None if not found
        """
        try:
            # Get existing document
            db_document = db.query(KnowledgeBase).filter(
                and_(
                    KnowledgeBase.id == document_id,
                    KnowledgeBase.organization_id == organization_id
                )
            ).first()
            
            if not db_document:
                return None
            
            # Update database record
            for field, value in update_data.items():
                if hasattr(db_document, field) and value is not None:
                    setattr(db_document, field, value)
            
            # Update vector database if content changed
            if "content" in update_data:
                await self._process_and_store_document(
                    db_document,
                    update_data.get("chunk_size", 1000),
                    update_data.get("chunk_overlap", 200)
                )
            
            db.commit()
            db.refresh(db_document)
            
            logger.info(f"Updated knowledge base document: {document_id}")
            return db_document
            
        except Exception as e:
            logger.error(f"Error updating knowledge base document: {e}")
            db.rollback()
            raise
    
    async def delete_document(self, db: Session, document_id: str, 
                            organization_id: str) -> bool:
        """
        Delete a knowledge base document.
        
        Args:
            db: Database session
            document_id: Document ID
            organization_id: Organization ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # Get document
            db_document = db.query(KnowledgeBase).filter(
                and_(
                    KnowledgeBase.id == document_id,
                    KnowledgeBase.organization_id == organization_id
                )
            ).first()
            
            if not db_document:
                return False
            
            # Delete from vector database (this will be handled by the vector service)
            # For now, we'll mark as inactive
            db_document.is_active = False
            db.commit()
            
            logger.info(f"Deleted knowledge base document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge base document: {e}")
            db.rollback()
            raise
    
    async def search_documents(self, organization_id: str, query: str,
                             limit: int = 10, score_threshold: float = 0.7,
                             assistant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search knowledge base documents using semantic search.
        
        Args:
            organization_id: Organization ID
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            assistant_id: Optional assistant ID filter
            
        Returns:
            List of search results
        """
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Search vector database
            results = await self.vector_service.search_similar(
                query_embedding=query_embedding,
                organization_id=organization_id,
                limit=limit,
                score_threshold=score_threshold,
                assistant_id=assistant_id
            )
            
            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Add search metadata
            for result in results:
                result["search_time_ms"] = search_time
            
            logger.info(f"Found {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            raise
    
    async def get_document(self, db: Session, document_id: str, 
                          organization_id: str) -> Optional[KnowledgeBase]:
        """
        Get a knowledge base document by ID.
        
        Args:
            db: Database session
            document_id: Document ID
            organization_id: Organization ID
            
        Returns:
            Document or None if not found
        """
        try:
            return db.query(KnowledgeBase).filter(
                and_(
                    KnowledgeBase.id == document_id,
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_active == True
                )
            ).first()
            
        except Exception as e:
            logger.error(f"Error getting knowledge base document: {e}")
            raise
    
    async def list_documents(self, db: Session, organization_id: str,
                           assistant_id: Optional[str] = None,
                           skip: int = 0, limit: int = 100) -> Tuple[List[KnowledgeBase], int]:
        """
        List knowledge base documents.
        
        Args:
            db: Database session
            organization_id: Organization ID
            assistant_id: Optional assistant ID filter
            skip: Number of records to skip
            limit: Maximum number of records
            
        Returns:
            Tuple of (documents, total_count)
        """
        try:
            query = db.query(KnowledgeBase).filter(
                and_(
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_active == True
                )
            )
            
            if assistant_id:
                query = query.filter(KnowledgeBase.assistant_id == assistant_id)
            
            total = query.count()
            documents = query.offset(skip).limit(limit).all()
            
            # Convert to dictionaries
            document_dicts = [doc.to_dict() for doc in documents]
            
            return document_dicts, total
            
        except Exception as e:
            logger.error(f"Error listing knowledge base documents: {e}")
            raise
    
    async def bulk_upload_documents(self, db: Session, organization_id: str,
                                  documents: List[Dict[str, Any]]) -> List[KnowledgeBase]:
        """
        Bulk upload multiple documents.
        
        Args:
            db: Database session
            organization_id: Organization ID
            documents: List of document data
            
        Returns:
            List of created documents
        """
        try:
            created_documents = []
            
            for doc_data in documents:
                doc_data["organization_id"] = organization_id
                document = await self.create_document(db, organization_id, doc_data)
                created_documents.append(document)
            
            logger.info(f"Bulk uploaded {len(documents)} documents")
            return created_documents
            
        except Exception as e:
            logger.error(f"Error bulk uploading documents: {e}")
            db.rollback()
            raise
    
    async def get_knowledge_stats(self, db: Session, organization_id: str) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Args:
            db: Database session
            organization_id: Organization ID
            
        Returns:
            Statistics dictionary
        """
        try:
            # Get document count
            total_documents = db.query(KnowledgeBase).filter(
                and_(
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_active == True
                )
            ).count()
            
            # Get assistant stats
            assistant_stats = db.query(
                Assistant.id,
                Assistant.name,
                db.query(KnowledgeBase).filter(
                    KnowledgeBase.assistant_id == Assistant.id,
                    KnowledgeBase.organization_id == organization_id,
                    KnowledgeBase.is_active == True
                ).count().label('document_count')
            ).filter(Assistant.organization_id == organization_id).all()
            
            # Get collection info from vector database
            collection_info = await self.vector_service.get_collection_info()
            
            return {
                "total_documents": total_documents,
                "total_chunks": collection_info.get("points_count", 0),
                "organization_id": organization_id,
                "assistant_stats": [
                    {
                        "assistant_id": str(stat.id),
                        "assistant_name": stat.name,
                        "document_count": stat.document_count
                    }
                    for stat in assistant_stats
                ],
                "vector_db_info": collection_info
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            raise
    
    async def _process_and_store_document(self, document: KnowledgeBase, 
                                        chunk_size: int, chunk_overlap: int):
        """
        Process document content and store in vector database.
        
        Args:
            document: Knowledge base document
            chunk_size: Chunk size for processing
            chunk_overlap: Chunk overlap
        """
        try:
            # Chunk the content
            chunks = self.embedding_service.chunk_text(
                document.content, 
                chunk_size=chunk_size, 
                overlap=chunk_overlap
            )
            
            # Generate embeddings for chunks
            embeddings = await self.embedding_service.generate_embeddings_batch(chunks)
            
            # Prepare documents for vector storage
            vector_documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_documents.append({
                    "content": chunk,
                    "title": f"{document.title} - Chunk {i+1}",
                    "organization_id": str(document.organization_id),
                    "assistant_id": str(document.assistant_id) if document.assistant_id else None,
                    "source": document.source,
                    "metadata": {
                        **(document.document_metadata or {}),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "document_id": str(document.id)
                    },
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat()
                })
            
            # Store in vector database
            await self.vector_service.add_documents(vector_documents, embeddings)
            
            logger.info(f"Processed and stored {len(chunks)} chunks for document: {document.id}")
            
        except Exception as e:
            logger.error(f"Error processing document for vector storage: {e}")
            raise


# Global instance
knowledge_service = KnowledgeService()
