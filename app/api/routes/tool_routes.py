"""
API routes for tool management and execution.
"""

import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud import tool_crud
from app.core.auth import get_current_user
from app.models.tool_schemas import (
    ToolCreate, ToolUpdate, ToolResponse, 
    AssistantToolCreate, AssistantToolUpdate, AssistantToolResponse,
    ToolExecutionRequest, ToolExecutionResponse
)
from app.db.models import User
from app.services.tool_execution_service import tool_execution_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tools"])


@router.post("/organizations/{organization_id}/tools", response_model=ToolResponse)
async def create_tool(
    organization_id: uuid.UUID,
    tool_data: ToolCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new tool for an organization."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    try:
        tool = tool_crud.create_tool(db, tool_data, organization_id)
        logger.info(f"Created tool: {tool.name} for organization: {organization_id}")
        return tool
    except Exception as e:
        logger.error(f"Error creating tool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create tool"
        )


@router.get("/organizations/{organization_id}/tools", response_model=List[ToolResponse])
async def get_tools(
    organization_id: uuid.UUID,
    skip: int = Query(0, ge=0, description="Number of tools to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of tools to return"),
    search: Optional[str] = Query(None, description="Search term for tool name or description"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get tools for an organization."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    try:
        tools = tool_crud.get_tools(
            db, organization_id, skip=skip, limit=limit, 
            search=search, is_active=is_active
        )
        return tools
    except Exception as e:
        logger.error(f"Error getting tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get tools"
        )


@router.get("/organizations/{organization_id}/tools/{tool_id}", response_model=ToolResponse)
async def get_tool(
    organization_id: uuid.UUID,
    tool_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific tool."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    tool = tool_crud.get_tool(db, tool_id, organization_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    
    return tool


@router.put("/organizations/{organization_id}/tools/{tool_id}", response_model=ToolResponse)
async def update_tool(
    organization_id: uuid.UUID,
    tool_id: uuid.UUID,
    tool_data: ToolUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a tool."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    tool = tool_crud.update_tool(db, tool_id, organization_id, tool_data)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    
    logger.info(f"Updated tool: {tool.name} for organization: {organization_id}")
    return tool


@router.delete("/organizations/{organization_id}/tools/{tool_id}")
async def delete_tool(
    organization_id: uuid.UUID,
    tool_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a tool."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    success = tool_crud.delete_tool(db, tool_id, organization_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    
    logger.info(f"Deleted tool: {tool_id} for organization: {organization_id}")
    return {"message": "Tool deleted successfully"}


@router.post("/organizations/{organization_id}/assistants/{assistant_id}/tools", response_model=AssistantToolResponse)
async def add_tool_to_assistant(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    tool_data: AssistantToolCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a tool to an assistant."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    assistant_tool = tool_crud.add_tool_to_assistant(
        db, assistant_id, tool_data, organization_id
    )
    if not assistant_tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant or tool not found"
        )
    
    logger.info(f"Added tool {tool_data.tool_id} to assistant {assistant_id}")
    return assistant_tool


@router.get("/organizations/{organization_id}/assistants/{assistant_id}/tools", response_model=List[AssistantToolResponse])
async def get_assistant_tools(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    enabled_only: bool = Query(True, description="Return only enabled tools"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get tools for an assistant."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    try:
        assistant_tools = tool_crud.get_assistant_tools(
            db, assistant_id, organization_id, enabled_only=enabled_only
        )
        return assistant_tools
    except Exception as e:
        logger.error(f"Error getting assistant tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get assistant tools"
        )


@router.put("/organizations/{organization_id}/assistant-tools/{assistant_tool_id}", response_model=AssistantToolResponse)
async def update_assistant_tool(
    organization_id: uuid.UUID,
    assistant_tool_id: uuid.UUID,
    tool_data: AssistantToolUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update assistant tool configuration."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    assistant_tool = tool_crud.update_assistant_tool(
        db, assistant_tool_id, organization_id, tool_data
    )
    if not assistant_tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant tool not found"
        )
    
    logger.info(f"Updated assistant tool: {assistant_tool_id}")
    return assistant_tool


@router.delete("/organizations/{organization_id}/assistant-tools/{assistant_tool_id}")
async def remove_tool_from_assistant(
    organization_id: uuid.UUID,
    assistant_tool_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a tool from an assistant."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    success = tool_crud.remove_tool_from_assistant(
        db, assistant_tool_id, organization_id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assistant tool not found"
        )
    
    logger.info(f"Removed assistant tool: {assistant_tool_id}")
    return {"message": "Tool removed from assistant successfully"}


@router.get("/organizations/{organization_id}/assistants/{assistant_id}/available-tools", response_model=List[ToolResponse])
async def get_available_tools_for_assistant(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get available tools that can be added to an assistant."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    try:
        available_tools = tool_crud.get_available_tools_for_assistant(
            db, assistant_id, organization_id
        )
        return available_tools
    except Exception as e:
        logger.error(f"Error getting available tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available tools"
        )


@router.post("/organizations/{organization_id}/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    organization_id: uuid.UUID,
    execution_request: ToolExecutionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute a tool (for testing purposes)."""
    # Verify user has access to organization
    if str(current_user.organization_id) != str(organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    # Get the tool
    tool = tool_crud.get_tool(db, execution_request.tool_id, organization_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool not found"
        )
    
    if not tool.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tool is not active"
        )
    
    # Execute the tool using the tool execution service
    try:
        result = await tool_execution_service.execute_tool(
            tool=tool,
            parameters=execution_request.parameters,
            body=execution_request.body
        )
        
        logger.info(f"Tool '{tool.name}' executed successfully for user {current_user.id}")
        return result
        
    except Exception as e:
        logger.error(f"Tool execution failed for tool '{tool.name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )
