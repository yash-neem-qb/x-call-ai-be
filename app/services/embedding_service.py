"""
Embedding service for generating vector embeddings from text.
Uses OpenAI embeddings exclusively.
"""

import logging
import asyncio
from typing import List, Optional
import numpy as np
from openai import AsyncOpenAI
import tiktoken

from app.config.vector_config import vector_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using OpenAI."""
    
    def __init__(self):
        self.openai_client: Optional[AsyncOpenAI] = None
        self.encoding = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize OpenAI embedding client."""
        try:
            if not vector_config.openai_api_key:
                raise ValueError("OpenAI API key is required for embedding service")
            
            # Initialize OpenAI client with minimal configuration
            self.openai_client = AsyncOpenAI(
                api_key=vector_config.openai_api_key
            )
            logger.info("OpenAI embedding client initialized")
            
            # Initialize tokenizer for chunking
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI embedding client: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding
        """
        try:
            return await self._generate_openai_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using OpenAI.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            return await self._generate_openai_embeddings_batch(texts)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = await self.openai_client.embeddings.create(
                model=vector_config.openai_embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API in batch."""
        try:
            # OpenAI supports batch processing
            response = await self.openai_client.embeddings.create(
                model=vector_config.openai_embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
    
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, 
                   overlap: Optional[int] = None) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = vector_config.chunk_size
        if overlap is None:
            overlap = vector_config.chunk_overlap
        
        try:
            # Tokenize the text
            tokens = self.encoding.encode(text)
            
            if len(tokens) <= chunk_size:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                
                # Decode tokens back to text
                chunk_text = self.encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                # Move start position: advance by (chunk_size - overlap)
                # This ensures proper overlap without infinite loops
                start += (chunk_size - overlap)
                
                # Safety check: if we've processed all tokens, break
                if start >= len(tokens):
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            # Fallback to simple character-based chunking
            return self._simple_chunk_text(text, chunk_size)
    
    def _simple_chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Simple character-based chunking fallback."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return 1536  # OpenAI text-embedding-3-small


# Global instance
embedding_service = EmbeddingService()
