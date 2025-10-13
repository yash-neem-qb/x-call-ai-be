"""
Tool execution service for making HTTP calls to external APIs.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
import aiohttp
from urllib.parse import urlencode, urlparse, parse_qs

from app.db.models import Tool, AssistantTool
from app.models.tool_schemas import ToolExecutionResponse

logger = logging.getLogger(__name__)


class ToolExecutionService:
    """Service for executing tool calls to external APIs."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize the HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes max
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("Tool execution service initialized")
    
    async def cleanup(self):
        """Clean up the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Tool execution service cleaned up")
    
    async def execute_tool(
        self, 
        tool: Tool, 
        parameters: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResponse:
        """
        Execute a tool call to an external API.
        
        Args:
            tool: The tool configuration
            parameters: URL parameters for the request
            body: Request body for POST/PUT requests
            
        Returns:
            ToolExecutionResponse with the result
        """
        if not self.session:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare URL with parameters
            url = self._prepare_url(tool, parameters)
            
            # Prepare headers
            headers = self._prepare_headers(tool)
            
            # Prepare request body
            request_body = self._prepare_body(tool, body)
            
            # Make the HTTP request with retries
            response_data, status_code = await self._make_request_with_retries(
                tool, url, headers, request_body
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.info(f"Tool '{tool.name}' executed successfully in {execution_time:.1f}ms")
            
            return ToolExecutionResponse(
                success=True,
                data=response_data,
                status_code=status_code,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool '{tool.name}' execution failed: {e}")
            
            return ToolExecutionResponse(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _prepare_url(self, tool: Tool, parameters: Optional[Dict[str, Any]]) -> str:
        """Prepare the URL with parameters."""
        url = tool.url
        
        # Add URL parameters from tool configuration
        if tool.parameters:
            tool_params = {}
            for param in tool.parameters:
                if isinstance(param, dict):
                    param_name = param.get('name')
                    param_value = param.get('default')
                    if param_name and param_value is not None:
                        tool_params[param_name] = param_value
            
            if tool_params:
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                query_params.update({k: [str(v)] for k, v in tool_params.items()})
                url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(query_params, doseq=True)}"
        
        # Add runtime parameters
        if parameters:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            query_params.update({k: [str(v)] for k, v in parameters.items()})
            url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(query_params, doseq=True)}"
        
        return url
    
    def _prepare_headers(self, tool: Tool) -> Dict[str, str]:
        """Prepare request headers."""
        headers = {
            'User-Agent': 'X-Call-AI-Tool-Execution/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Add custom headers from tool configuration
        if tool.headers:
            headers.update(tool.headers)
        
        return headers
    
    def _prepare_body(self, tool: Tool, body: Optional[Dict[str, Any]]) -> Optional[str]:
        """Prepare request body for POST/PUT requests."""
        if tool.method.upper() in ['GET', 'DELETE']:
            return None
        
        # Use provided body or tool's body schema
        request_body = body or tool.body_schema
        
        if request_body:
            return json.dumps(request_body)
        
        return None
    
    async def _make_request_with_retries(
        self, 
        tool: Tool, 
        url: str, 
        headers: Dict[str, str], 
        body: Optional[str]
    ) -> tuple[Any, int]:
        """Make HTTP request with retry logic."""
        last_exception = None
        
        for attempt in range(tool.retry_count + 1):
            try:
                async with self.session.request(
                    method=tool.method,
                    url=url,
                    headers=headers,
                    data=body
                ) as response:
                    
                    # Read response text
                    response_text = await response.text()
                    
                    # Try to parse as JSON
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = response_text
                    
                    # Check if response is successful
                    if response.status < 400:
                        return response_data, response.status
                    else:
                        # For HTTP errors, don't retry
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}: {response_text}"
                        )
            
            except Exception as e:
                last_exception = e
                if attempt < tool.retry_count:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Tool '{tool.name}' attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Tool '{tool.name}' failed after {tool.retry_count + 1} attempts: {e}")
        
        # If we get here, all retries failed
        raise last_exception or Exception("Tool execution failed after all retries")
    
    async def execute_tools_for_assistant(
        self, 
        assistant_tools: List[AssistantTool], 
        context: Dict[str, Any]
    ) -> List[ToolExecutionResponse]:
        """
        Execute multiple tools for an assistant based on context.
        
        Args:
            assistant_tools: List of tools assigned to the assistant
            context: Context information for tool execution
            
        Returns:
            List of tool execution responses
        """
        if not assistant_tools:
            return []
        
        # Filter enabled tools and sort by priority
        enabled_tools = [
            at for at in assistant_tools 
            if at.is_enabled and at.tool.is_active
        ]
        enabled_tools.sort(key=lambda x: x.priority, reverse=True)
        
        # Execute tools concurrently
        tasks = []
        for assistant_tool in enabled_tools:
            task = self.execute_tool(
                tool=assistant_tool.tool,
                parameters=context.get('parameters'),
                body=context.get('body')
            )
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error responses
            execution_responses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    execution_responses.append(ToolExecutionResponse(
                        success=False,
                        error=str(result),
                        execution_time_ms=0.0
                    ))
                else:
                    execution_responses.append(result)
            
            return execution_responses
        
        return []


# Global instance
tool_execution_service = ToolExecutionService()
