"""
LLM service using OpenAI.
Provides a clean interface for generating AI responses.
"""

import json
import logging
import aiohttp
from typing import Optional, Dict, Any, Callable, Coroutine, List

from app.config.settings import settings

logger = logging.getLogger(__name__)


class OpenAILLMService:
    """Service for LLM operations using OpenAI API."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.api_key = settings.openai_api_key
        # Defaults from settings
        self.model = getattr(settings, "openai_chat_model", "gpt-4o")
        self.max_tokens = getattr(settings, "openai_max_tokens", 256)
        self.temperature = getattr(settings, "openai_temperature", 0.8)
        self.system_prompt = getattr(
            settings,
            "openai_system_prompt",
            "You are a helpful voice assistant. Be concise and conversational.",
        )
    async def initialize(self):
        """
        Initialize the LLM service.
        This is a no-op for OpenAI as connections are per-request.
        """
        logger.info("LLM service initialized")
        return True
    
    async def generate_response(self, 
                               text: str, 
                               on_content_delta: Callable[[str], Coroutine],
                               conversation_history: List[Dict[str, Any]] = None,
                               custom_system_prompt: str = None,
                               model: str = None,
                               should_stop: Optional[Callable[[], bool]] = None) -> str:
        """
        Generate an AI response to the given text.
        
        Args:
            text: The input text to respond to
            on_content_delta: Callback for content chunks
            conversation_history: Optional conversation history
            custom_system_prompt: Optional custom system prompt
            
        Returns:
            str: The complete generated response
        """
        try:
            # Initialize variables
            response_text = ""
            chunk_count = 0
            
            # Prepare messages
            messages = []
            
            # Add system message
            system_message = custom_system_prompt if custom_system_prompt else self.system_prompt
            messages.append({"role": "system", "content": system_message})
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add user message
            messages.append({"role": "user", "content": text})
            
            # Create async HTTP client session
            async with aiohttp.ClientSession() as session:
                # Prepare request
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                # Use provided model or fall back to default
                model_name = model or self.model
                logger.info(f"Using LLM model: {model_name}")
                
                data = {
                    "model": model_name,
                    "messages": messages,
                    "stream": True
                }
                
                # Use the appropriate parameters based on the model
                lower_model = model_name.lower()
                # Newer models (gpt-5 family) use max_completion_tokens and fixed temperature
                if "gpt-5" in lower_model:
                    data["max_completion_tokens"] = self.max_tokens
                    # According to OpenAI docs, temperature is fixed at 1 for gpt-5
                    logger.info("Using default temperature=1 for gpt-5 model")
                else:
                    data["max_tokens"] = self.max_tokens
                    data["temperature"] = self.temperature
                
                # Make streaming request
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        return "I'm sorry, I couldn't process that right now."
                    
                    # Process streaming response
                    async for line in response.content:
                        if should_stop and should_stop():
                            logger.info("LLM streaming aborted due to barge-in")
                            break
                        line = line.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue
                            
                        if line.startswith("data: "):
                            chunk_count += 1
                            json_str = line[6:]  # Remove "data: " prefix
                            try:
                                chunk = json.loads(json_str)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                                    if delta:
                                        response_text += delta
                                        # Process chunk through callback (removed console printing for performance)
                                        await on_content_delta(delta)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON from chunk: {json_str}")
            
            # Removed console printing for performance
            logger.info(f"Response generated: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, there was an error processing your request."


# Global OpenAI LLM service instance
openai_llm_service = OpenAILLMService()
