"""
Vector database service using Qdrant for knowledge base operations.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config.vector_config import vector_config

logger = logging.getLogger(__name__)


class VectorService:
    """Service for vector database operations."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name = vector_config.knowledge_collection_name
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            if vector_config.qdrant_url:
                # Cloud Qdrant
                self.client = QdrantClient(
                    url=vector_config.qdrant_url,
                    api_key=vector_config.qdrant_api_key
                )
                logger.info("Connected to cloud Qdrant")
            else:
                # Local Qdrant
                self.client = QdrantClient(
                    host=vector_config.qdrant_host,
                    port=vector_config.qdrant_port
                )
                logger.info(f"Connected to local Qdrant at {vector_config.qdrant_host}:{vector_config.qdrant_port}")
            
            # Ensure collection exists
            self._ensure_collection_exists()
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the knowledge base collection exists with proper indexes."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self._create_collection()
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                # Ensure indexes exist even for existing collections
                self._ensure_indexes_exist()
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def _ensure_indexes_exist(self):
        """Ensure indexes exist for existing collections."""
        try:
            # Try to create indexes - they may already exist
            self._create_indexes()
        except Exception as e:
            logger.warning(f"Could not create indexes (may already exist): {e}")
    
    def _create_collection(self):
        """Create the knowledge base collection with proper indexing."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_config.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create indexes for filtering
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for efficient filtering."""
        try:
            # Create index for organization_id (required for filtering)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="organization_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index for organization_id in collection {self.collection_name}")
            
            # Create index for assistant_id (optional filtering)
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="assistant_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Created index for assistant_id in collection {self.collection_name}")
            
        except Exception as e:
            logger.warning(f"Error creating indexes (collection may already exist): {e}")
            # Don't raise here as the collection might already exist with indexes
    
    async def add_documents(self, documents: List[Dict[str, Any]], 
                          embeddings: List[List[float]]) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document metadata
            embeddings: List of corresponding embeddings
            
        Returns:
            List of document IDs
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            points = []
            document_ids = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": doc.get("content", ""),
                        "title": doc.get("title", ""),
                        "organization_id": doc.get("organization_id"),
                        "assistant_id": doc.get("assistant_id"),
                        "source": doc.get("source", ""),
                        "document_metadata": doc.get("metadata", {}),
                        "created_at": doc.get("created_at"),
                        "updated_at": doc.get("updated_at")
                    }
                )
                points.append(point)
            
            # Insert points in batches
            batch_size = 10
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            logger.info(f"Added {len(documents)} documents to vector database")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    async def search_similar(self, query_embedding: List[float], 
                           organization_id: str,
                           limit: Optional[int] = None,
                           score_threshold: Optional[float] = None,
                           assistant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            organization_id: Organization ID to filter by
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            assistant_id: Optional assistant ID to filter by
            
        Returns:
            List of similar documents with scores
        """
        try:
            if limit is None:
                limit = vector_config.default_search_limit
            if score_threshold is None:
                score_threshold = vector_config.similarity_threshold
            
            # Build filter
            filter_conditions = [
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=organization_id)
                )
            ]
            
            if assistant_id:
                filter_conditions.append(
                    FieldCondition(
                        key="assistant_id",
                        match=MatchValue(value=assistant_id)
                    )
                )
            
            search_filter = Filter(must=filter_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "title": result.payload.get("title", ""),
                    "source": result.payload.get("source", ""),
                    "metadata": result.payload.get("document_metadata", {}),
                    "created_at": result.payload.get("created_at"),
                    "updated_at": result.payload.get("updated_at")
                })
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector database.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=document_ids)
            )
            
            logger.info(f"Deleted {len(document_ids)} documents from vector database")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    async def update_document(self, document_id: str, 
                            content: str, 
                            embedding: List[float],
                            metadata: Dict[str, Any]) -> bool:
        """
        Update a document in the vector database.
        
        Args:
            document_id: Document ID to update
            content: New content
            embedding: New embedding
            metadata: Updated metadata
            
        Returns:
            True if successful
        """
        try:
            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload={
                    "content": content,
                    "title": metadata.get("title", ""),
                    "organization_id": metadata.get("organization_id"),
                    "assistant_id": metadata.get("assistant_id"),
                    "source": metadata.get("source", ""),
                    "document_metadata": metadata.get("metadata", {}),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at")
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Updated document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.params.vectors.size,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if the vector database is healthy."""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector database health check failed: {e}")
            return False


# Global instance
vector_service = VectorService()
