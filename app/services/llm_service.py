"""
LLM service using OpenAI.
Provides a clean interface for generating AI responses.
"""

import json
import logging
import aiohttp
import openai
from typing import Optional, Dict, Any, Callable, Coroutine, List, Union
from dataclasses import dataclass

from app.config.settings import settings
from app.db.models import AssistantTool
from app.services.tool_execution_service import tool_execution_service
from app.services.end_call_service import end_call_service

logger = logging.getLogger(__name__)


@dataclass
class EndCallResult:
    """Result when user requests to end the call."""
    should_end: bool = True
    response: str = ""
    end_reason: str = "call_end_requested"
    metadata: Dict[str, Any] = None


class OpenAILLMService:
    """Service for LLM operations using OpenAI API."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.api_key = settings.openai_api_key
        # No hardcoded defaults - everything comes from database
        self.client = None
    async def initialize(self):
        """
        Initialize the LLM service.
        Creates an OpenAI client instance for other services to use.
        """
        import openai
        self.client = openai.AsyncClient(api_key=self.api_key)
        logger.info("LLM service initialized")
        return True
    
    async def generate_response(self, 
                               text: str, 
                               on_content_delta: Callable[[str], Coroutine],
                               conversation_history: List[Dict[str, Any]] = None,
                               custom_system_prompt: str = None,
                               model: str = None,
                               max_tokens: int = None,
                               temperature: float = None,
                               should_stop: Optional[Callable[[], bool]] = None) -> Union[str, EndCallResult]:
        """
        Generate an AI response to the given text.
        
        Args:
            text: The input text to respond to
            on_content_delta: Callback for content chunks
            conversation_history: Optional conversation history
            custom_system_prompt: Optional custom system prompt
            model: LLM model to use (required)
            max_tokens: Maximum tokens for response (required)
            temperature: Temperature for response generation (required)
            
        Returns:
            Union[str, EndCallResult]: The complete generated response or end call result
        """
        try:
            # Check for end call request first
            if end_call_service.should_end_call(text):
                logger.info(f"End call request detected: '{text[:50]}...'")
                end_response = end_call_service.get_end_call_response(text)
                
                # Send the goodbye response
                if on_content_delta:
                    await on_content_delta(end_response)
                
                return EndCallResult(
                    should_end=True,
                    response=end_response,
                    end_reason=end_call_service.end_reason,
                    metadata=end_call_service.get_end_call_metadata()
                )
            
            # Validate required parameters
            if not model:
                raise ValueError("Model is required")
            if max_tokens is None:
                raise ValueError("Max tokens is required")
            if temperature is None:
                raise ValueError("Temperature is required")
            if not custom_system_prompt:
                raise ValueError("System prompt is required")
            
            # Initialize variables
            response_text = ""
            chunk_count = 0
            
            # Prepare messages
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": custom_system_prompt})
            
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
                logger.info(f"Using LLM model: {model}")
                
                data = {
                    "model": model,
                    "messages": messages,
                    "stream": True
                }
                
                # Use the appropriate parameters based on the model
                lower_model = model.lower()
                # Newer models (gpt-5 family) use max_completion_tokens and fixed temperature
                if "gpt-5" in lower_model:
                    data["max_completion_tokens"] = max_tokens
                    # According to OpenAI docs, temperature is fixed at 1 for gpt-5
                    logger.info("Using default temperature=1 for gpt-5 model")
                else:
                    data["max_tokens"] = max_tokens
                    data["temperature"] = temperature
                
                # Make streaming request
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                    
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
            raise e
    
    async def generate_response_with_tools(
        self,
        text: str,
        on_content_delta: Callable[[str], Coroutine],
        assistant_tools: List[AssistantTool],
        conversation_history: List[Dict[str, Any]] = None,
        custom_system_prompt: str = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        should_stop: Optional[Callable[[], bool]] = None
    ) -> Union[str, EndCallResult]:
        """
        Generate an AI response with tool calling capabilities.
        
        Args:
            text: The input text to respond to
            on_content_delta: Callback for content chunks
            assistant_tools: List of tools available to the assistant
            conversation_history: Optional conversation history
            custom_system_prompt: Optional custom system prompt
            model: LLM model to use (required)
            max_tokens: Maximum tokens for response (required)
            temperature: Temperature for response generation (required)
            should_stop: Optional callback to check if generation should stop
            
        Returns:
            Union[str, EndCallResult]: The complete generated response or end call result
        """
        try:
            # Check for end call request first
            if end_call_service.should_end_call(text):
                logger.info(f"End call request detected in tool-enabled response: '{text[:50]}...'")
                end_response = end_call_service.get_end_call_response(text)
                
                # Send the goodbye response
                if on_content_delta:
                    await on_content_delta(end_response)
                
                return EndCallResult(
                    should_end=True,
                    response=end_response,
                    end_reason=end_call_service.end_reason,
                    metadata=end_call_service.get_end_call_metadata()
                )
            
            # Validate required parameters
            if not model:
                raise ValueError("Model is required")
            if max_tokens is None:
                raise ValueError("Max tokens is required")
            if temperature is None:
                raise ValueError("Temperature is required")
            if not custom_system_prompt:
                raise ValueError("System prompt is required")
            
            # Initialize tool execution service if needed
            await tool_execution_service.initialize()
            
            # Prepare messages
            messages = []
            
            # Add system message with tool information
            system_message = self._build_system_prompt_with_tools(
                custom_system_prompt, assistant_tools
            )
            messages.append({"role": "system", "content": system_message})
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add user message
            messages.append({"role": "user", "content": text})
            
            # Prepare tools for OpenAI function calling
            tools = self._prepare_tools_for_openai(assistant_tools)
            
            # Create async HTTP client session
            async with aiohttp.ClientSession() as session:
                # Prepare request
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                logger.info(f"Using LLM model with tools: {model}")
                
                data = {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto"  # Let the model decide when to use tools
                }
                
                # Use the appropriate parameters based on the model
                lower_model = model.lower()
                if "gpt-5" in lower_model:
                    data["max_completion_tokens"] = max_tokens
                else:
                    data["max_tokens"] = max_tokens
                    data["temperature"] = temperature
                
                # Make request
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                    
                    response_data = await response.json()
                    message = response_data["choices"][0]["message"]
                    
                    # Check if the model wants to call tools
                    if message.get("tool_calls"):
                        # Execute tool calls
                        tool_results = await self._execute_tool_calls(
                            message["tool_calls"], assistant_tools
                        )
                        
                        # Add tool results to conversation
                        messages.append(message)  # Add assistant message with tool calls
                        for tool_result in tool_results:
                            messages.append(tool_result)
                        
                        # Generate final response
                        final_data = {
                            "model": model,
                            "messages": messages,
                            "stream": True
                        }
                        
                        if "gpt-5" in lower_model:
                            final_data["max_completion_tokens"] = max_tokens
                        else:
                            final_data["max_tokens"] = max_tokens
                            final_data["temperature"] = temperature
                        
                        # Make streaming request for final response
                        async with session.post(url, headers=headers, json=final_data) as final_response:
                            if final_response.status != 200:
                                error_text = await final_response.text()
                                logger.error(f"OpenAI API error: {final_response.status} - {error_text}")
                                raise Exception(f"OpenAI API error: {final_response.status} - {error_text}")
                            
                            # Process streaming response
                            response_text = ""
                            async for line in final_response.content:
                                if should_stop and should_stop():
                                    logger.info("LLM streaming aborted due to barge-in")
                                    break
                                line = line.decode('utf-8').strip()
                                if not line or line == "data: [DONE]":
                                    continue
                                    
                                if line.startswith("data: "):
                                    json_str = line[6:]  # Remove "data: " prefix
                                    try:
                                        chunk = json.loads(json_str)
                                        if "choices" in chunk and len(chunk["choices"]) > 0:
                                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                                            if delta:
                                                response_text += delta
                                                await on_content_delta(delta)
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse JSON from chunk: {json_str}")
                            
                            return response_text
                    else:
                        # No tool calls, return direct response
                        response_text = message.get("content", "")
                        await on_content_delta(response_text)
                        return response_text
            
        except Exception as e:
            logger.error(f"Error generating response with tools: {e}")
            raise e
    
    def _build_system_prompt_with_tools(
        self, 
        custom_system_prompt: str, 
        assistant_tools: List[AssistantTool]
    ) -> str:
        """Build system prompt with tool information."""
        base_prompt = custom_system_prompt
        
        if not assistant_tools:
            return base_prompt
        
        tool_descriptions = []
        for assistant_tool in assistant_tools:
            if assistant_tool.is_enabled and assistant_tool.tool.is_active:
                tool_desc = f"- {assistant_tool.tool.name}: {assistant_tool.tool.description or 'No description'}"
                tool_descriptions.append(tool_desc)
        
        if tool_descriptions:
            tools_text = "\n".join(tool_descriptions)
            return f"{base_prompt}\n\nYou have access to the following tools:\n{tools_text}\n\nUse these tools when appropriate to help the user."
        
        return base_prompt
    
    def _prepare_tools_for_openai(self, assistant_tools: List[AssistantTool]) -> List[Dict[str, Any]]:
        """Prepare tools in OpenAI function calling format."""
        tools = []
        
        for assistant_tool in assistant_tools:
            if not assistant_tool.is_enabled or not assistant_tool.tool.is_active:
                continue
            
            tool = assistant_tool.tool
            
            # Build function schema
            function_schema = {
                "type": "function",
                "function": {
                    "name": tool.name.lower().replace(" ", "_"),
                    "description": tool.description or f"Call {tool.name}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add parameters from tool configuration
            if tool.parameters:
                for param in tool.parameters:
                    if isinstance(param, dict):
                        param_name = param.get('name')
                        param_type = param.get('type', 'string')
                        param_desc = param.get('description', '')
                        param_required = param.get('required', False)
                        
                        if param_name:
                            function_schema["function"]["parameters"]["properties"][param_name] = {
                                "type": param_type,
                                "description": param_desc
                            }
                            
                            if param_required:
                                function_schema["function"]["parameters"]["required"].append(param_name)
            
            tools.append(function_schema)
        
        return tools
    
    async def _execute_tool_calls(
        self, 
        tool_calls: List[Dict[str, Any]], 
        assistant_tools: List[AssistantTool]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            # Find the corresponding tool
            assistant_tool = None
            for at in assistant_tools:
                if at.tool.name.lower().replace(" ", "_") == function_name:
                    assistant_tool = at
                    break
            
            if not assistant_tool:
                logger.warning(f"Tool not found: {function_name}")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"Tool '{function_name}' not found"
                })
                continue
            
            # Execute the tool
            try:
                result = await tool_execution_service.execute_tool(
                    tool=assistant_tool.tool,
                    parameters=arguments
                )
                
                if result.success:
                    content = f"Tool executed successfully: {json.dumps(result.data)}"
                else:
                    content = f"Tool execution failed: {result.error}"
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {function_name}: {e}")
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": f"Tool execution error: {str(e)}"
                })
        
        return tool_results


# Global OpenAI LLM service instance
openai_llm_service = OpenAILLMService()
