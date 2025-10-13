"""
Pydantic schemas for tool-related operations.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
import uuid


class ToolParameter(BaseModel):
    """Schema for tool parameters."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, etc.)")
    description: Optional[str] = Field(None, description="Parameter description")
    required: bool = Field(False, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")


class ToolHeader(BaseModel):
    """Schema for tool headers."""
    key: str = Field(..., description="Header key")
    value: str = Field(..., description="Header value")


class ToolCreate(BaseModel):
    """Schema for creating a new tool."""
    name: str = Field(..., min_length=1, max_length=255, description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    method: str = Field(..., description="HTTP method (GET, POST, PUT, DELETE, etc.)")
    url: str = Field(..., description="Tool endpoint URL")
    headers: Optional[List[ToolHeader]] = Field(None, description="Custom headers")
    parameters: Optional[List[ToolParameter]] = Field(None, description="URL parameters")
    body_schema: Optional[str] = Field(None, description="Request body schema as JSON string")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Request timeout in seconds")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retry attempts")


class ToolUpdate(BaseModel):
    """Schema for updating a tool."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    method: Optional[str] = Field(None, description="HTTP method")
    url: Optional[str] = Field(None, description="Tool endpoint URL")
    headers: Optional[List[ToolHeader]] = Field(None, description="Custom headers")
    parameters: Optional[List[ToolParameter]] = Field(None, description="URL parameters")
    body_schema: Optional[str] = Field(None, description="Request body schema as JSON string")
    timeout_seconds: Optional[int] = Field(None, ge=1, le=300, description="Request timeout in seconds")
    retry_count: Optional[int] = Field(None, ge=0, le=10, description="Number of retry attempts")
    is_active: Optional[bool] = Field(None, description="Whether tool is active")


class ToolResponse(BaseModel):
    """Schema for tool response."""
    id: uuid.UUID = Field(..., description="Tool ID")
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    method: str = Field(..., description="HTTP method")
    url: str = Field(..., description="Tool endpoint URL")
    headers: Optional[List[ToolHeader]] = Field(None, description="Custom headers")
    parameters: Optional[List[ToolParameter]] = Field(None, description="URL parameters")
    body_schema: Optional[str] = Field(None, description="Request body schema as JSON string")
    timeout_seconds: int = Field(..., description="Request timeout in seconds")
    retry_count: int = Field(..., description="Number of retry attempts")
    is_active: bool = Field(..., description="Whether tool is active")
    organization_id: uuid.UUID = Field(..., description="Organization ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class AssistantToolCreate(BaseModel):
    """Schema for adding a tool to an assistant."""
    tool_id: uuid.UUID = Field(..., description="Tool ID to add")
    is_enabled: bool = Field(True, description="Whether tool is enabled for this assistant")
    priority: int = Field(0, description="Tool priority (higher number = higher priority)")


class AssistantToolUpdate(BaseModel):
    """Schema for updating assistant tool configuration."""
    is_enabled: Optional[bool] = Field(None, description="Whether tool is enabled")
    priority: Optional[int] = Field(None, description="Tool priority")


class AssistantToolResponse(BaseModel):
    """Schema for assistant tool response."""
    id: uuid.UUID = Field(..., description="Assistant tool ID")
    assistant_id: uuid.UUID = Field(..., description="Assistant ID")
    tool_id: uuid.UUID = Field(..., description="Tool ID")
    is_enabled: bool = Field(..., description="Whether tool is enabled")
    priority: int = Field(..., description="Tool priority")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    tool: ToolResponse = Field(..., description="Tool details")
    
    class Config:
        from_attributes = True


class ToolExecutionRequest(BaseModel):
    """Schema for tool execution request."""
    tool_id: uuid.UUID = Field(..., description="Tool ID to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters")
    body: Optional[Dict[str, Any]] = Field(None, description="Request body for POST/PUT")


class ToolExecutionResponse(BaseModel):
    """Schema for tool execution response."""
    success: bool = Field(..., description="Whether execution was successful")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
