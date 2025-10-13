"""
RAG-Enhanced LLM Service.
Integrates knowledge base retrieval with LLM response generation for intelligent call handling.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable, Coroutine
from dataclasses import dataclass

from app.services.llm_service import openai_llm_service
from app.services.knowledge_service import knowledge_service
from app.db.crud.tool_crud import get_assistant_tools
from app.db.database import get_db
from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval-Augmented Generation) behavior."""
    
    # Knowledge base search settings
    enable_rag: bool = False  # Disabled for testing to reduce latency
    max_knowledge_results: int = 3
    knowledge_score_threshold: float = 0.7
    max_knowledge_context_length: int = 2000
    
    # System prompt enhancement
    include_knowledge_in_system_prompt: bool = True
    knowledge_context_template: str = """
Based on the following knowledge base information, provide accurate and helpful responses:

KNOWLEDGE BASE CONTEXT:
{knowledge_context}

Use this information to answer questions accurately. If the knowledge base doesn't contain relevant information, say so clearly and provide general assistance based on your training.
"""
    
    # Fallback behavior
    fallback_to_general_llm: bool = True
    log_knowledge_usage: bool = True


class RAGEnhancedLLMService:
    """
    RAG-enhanced LLM service that searches knowledge base before generating responses.
    
    This service combines the power of semantic search with LLM generation to provide
    accurate, context-aware responses during voice calls.
    """
    
    def __init__(self, rag_config: Optional[RAGConfig] = None):
        """
        Initialize the RAG-enhanced LLM service.
        
        Args:
            rag_config: Configuration for RAG behavior
        """
        self.llm_service = openai_llm_service
        self.knowledge_service = knowledge_service
        self.config = rag_config or RAGConfig()
        
        logger.info("RAG-enhanced LLM service initialized")
    
    async def initialize(self):
        """
        Initialize the RAG-enhanced LLM service.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize underlying services
            await self.llm_service.initialize()
            logger.info("RAG-enhanced LLM service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG-enhanced LLM service: {e}")
            return False
    
    async def generate_response_with_rag(
        self,
        text: str,
        on_content_delta: Callable[[str], Coroutine],
        conversation_history: List[Dict[str, Any]] = None,
        custom_system_prompt: str = None,
        model: str = None,
        organization_id: str = None,
        assistant_id: str = None,
        should_stop: Optional[Callable[[], bool]] = None
    ) -> str:
        """
        Generate an AI response using RAG (Retrieval-Augmented Generation).
        
        Args:
            text: The input text to respond to
            on_content_delta: Callback for content chunks
            conversation_history: Optional conversation history
            custom_system_prompt: Optional custom system prompt
            model: Optional model override
            organization_id: Organization ID for knowledge base access
            assistant_id: Assistant ID for knowledge base filtering
            should_stop: Optional callback to check if generation should stop
            
        Returns:
            str: The complete generated response
        """
        try:
            start_time = time.time()
            
            # Step 1: Search knowledge base if RAG is enabled
            knowledge_context = ""
            knowledge_results = []
            
            if self.config.enable_rag and organization_id:
                knowledge_context, knowledge_results = await self._search_knowledge_base(
                    text, organization_id, assistant_id
                )
            
            # Step 2: Enhance system prompt with knowledge context
            enhanced_system_prompt = self._build_enhanced_system_prompt(
                custom_system_prompt, knowledge_context
            )
            
            # Step 3: Get assistant tools if assistant_id is provided
            assistant_tools = []
            if assistant_id and organization_id:
                try:
                    db = next(get_db())
                    assistant_tools = get_assistant_tools(db, assistant_id, organization_id, enabled_only=True)
                    db.close()
                except Exception as e:
                    logger.warning(f"Failed to load assistant tools: {e}")
            
            # Step 4: Generate response using enhanced context and tools
            if assistant_tools:
                response = await self.llm_service.generate_response_with_tools(
                    text=text,
                    on_content_delta=on_content_delta,
                    assistant_tools=assistant_tools,
                    conversation_history=conversation_history,
                    custom_system_prompt=enhanced_system_prompt,
                    model=model,
                    should_stop=should_stop
                )
            else:
                response = await self.llm_service.generate_response(
                    text=text,
                    on_content_delta=on_content_delta,
                    conversation_history=conversation_history,
                    custom_system_prompt=enhanced_system_prompt,
                    model=model,
                    should_stop=should_stop
                )
            
            # Step 5: Log knowledge usage if enabled
            if self.config.log_knowledge_usage and knowledge_results:
                self._log_knowledge_usage(text, knowledge_results, response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG-enhanced response generation: {e}")
            
            # Fallback to general LLM if configured
            if self.config.fallback_to_general_llm:
                logger.info("Falling back to general LLM service")
                
                # Try to get tools for fallback as well
                assistant_tools = []
                if assistant_id and organization_id:
                    try:
                        db = next(get_db())
                        assistant_tools = get_assistant_tools(db, assistant_id, organization_id, enabled_only=True)
                        db.close()
                    except Exception as e:
                        logger.warning(f"Failed to load assistant tools for fallback: {e}")
                
                if assistant_tools:
                    return await self.llm_service.generate_response_with_tools(
                        text=text,
                        on_content_delta=on_content_delta,
                        assistant_tools=assistant_tools,
                        conversation_history=conversation_history,
                        custom_system_prompt=custom_system_prompt,
                        model=model,
                        should_stop=should_stop
                    )
                else:
                    return await self.llm_service.generate_response(
                        text=text,
                        on_content_delta=on_content_delta,
                        conversation_history=conversation_history,
                        custom_system_prompt=custom_system_prompt,
                        model=model,
                        should_stop=should_stop
                    )
            else:
                return "I'm sorry, I couldn't process that right now."
    
    async def _search_knowledge_base(
        self, 
        query: str, 
        organization_id: str, 
        assistant_id: Optional[str] = None
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            organization_id: Organization ID
            assistant_id: Optional assistant ID for filtering
            
        Returns:
            Tuple of (knowledge_context, search_results)
        """
        try:
            logger.info(f"Searching knowledge base for query: '{query[:50]}...'")
            
            # Search knowledge base
            results = await self.knowledge_service.search_documents(
                organization_id=organization_id,
                query=query,
                limit=self.config.max_knowledge_results,
                score_threshold=self.config.knowledge_score_threshold,
                assistant_id=assistant_id
            )
            
            if not results:
                logger.info("No relevant knowledge base results found")
                return "", []
            
            # Build knowledge context from results
            knowledge_context = self._build_knowledge_context(results)
            
            logger.info(f"Found {len(results)} relevant knowledge base results")
            return knowledge_context, results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return "", []
    
    def _build_knowledge_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Build knowledge context string from search results.
        
        Args:
            results: Knowledge base search results
            
        Returns:
            Formatted knowledge context string
        """
        if not results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Extract relevant information
            title = result.get("title", "Unknown Document")
            content = result.get("content", "")
            score = result.get("score", 0.0)
            source = result.get("source", "")
            
            # Truncate content if too long
            max_content_length = self.config.max_knowledge_context_length // len(results)
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            # Format the result
            context_part = f"Document {i}: {title}"
            if source:
                context_part += f" (Source: {source})"
            context_part += f"\nRelevance Score: {score:.2f}\nContent: {content}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_enhanced_system_prompt(
        self, 
        custom_system_prompt: Optional[str], 
        knowledge_context: str
    ) -> str:
        """
        Build enhanced system prompt with knowledge context.
        
        Args:
            custom_system_prompt: Original system prompt
            knowledge_context: Knowledge base context
            
        Returns:
            Enhanced system prompt
        """
        # Start with custom prompt or default
        base_prompt = custom_system_prompt or self.llm_service.system_prompt
        
        # Add knowledge context if available and configured
        if knowledge_context and self.config.include_knowledge_in_system_prompt:
            enhanced_prompt = base_prompt + "\n\n" + self.config.knowledge_context_template.format(
                knowledge_context=knowledge_context
            )
        else:
            enhanced_prompt = base_prompt
        
        return enhanced_prompt
    
    def _log_knowledge_usage(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        response: str, 
        processing_time: float
    ):
        """
        Log knowledge base usage for analytics and debugging.
        
        Args:
            query: Original query
            results: Knowledge base results used
            response: Generated response
            processing_time: Total processing time
        """
        try:
            # Extract relevant metrics
            result_count = len(results)
            avg_score = sum(r.get("score", 0) for r in results) / result_count if results else 0
            sources = [r.get("source", "unknown") for r in results]
            
            logger.info(
                f"RAG Usage - Query: '{query[:50]}...', "
                f"Results: {result_count}, "
                f"Avg Score: {avg_score:.2f}, "
                f"Sources: {sources}, "
                f"Processing Time: {processing_time:.2f}s, "
                f"Response Length: {len(response)} chars"
            )
            
        except Exception as e:
            logger.error(f"Error logging knowledge usage: {e}")
    
    def update_config(self, new_config: RAGConfig):
        """
        Update RAG configuration.
        
        Args:
            new_config: New RAG configuration
        """
        self.config = new_config
        logger.info("RAG configuration updated")
    
    def get_config(self) -> RAGConfig:
        """
        Get current RAG configuration.
        
        Returns:
            Current RAG configuration
        """
        return self.config


# Global RAG-enhanced LLM service instance
rag_llm_service = RAGEnhancedLLMService()
