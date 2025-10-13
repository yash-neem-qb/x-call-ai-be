"""
CRUD operations for tools and assistant tools.
"""

import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc
from datetime import datetime

from app.db.models import Tool, AssistantTool, Assistant
from app.models.tool_schemas import ToolCreate, ToolUpdate, AssistantToolCreate, AssistantToolUpdate


def create_tool(db: Session, tool_data: ToolCreate, organization_id: uuid.UUID) -> Tool:
    """Create a new tool."""
    tool = Tool(
        name=tool_data.name,
        description=tool_data.description,
        method=tool_data.method.upper(),
        url=tool_data.url,
        headers=[header.dict() for header in tool_data.headers] if tool_data.headers else None,
        parameters=[param.dict() for param in tool_data.parameters] if tool_data.parameters else None,
        body_schema=tool_data.body_schema,  # Already a string, no conversion needed
        timeout_seconds=tool_data.timeout_seconds,
        retry_count=tool_data.retry_count,
        organization_id=organization_id
    )
    
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return tool


def get_tool(db: Session, tool_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Tool]:
    """Get a tool by ID within an organization."""
    return db.query(Tool).filter(
        and_(
            Tool.id == tool_id,
            Tool.organization_id == organization_id
        )
    ).first()


def get_tools(
    db: Session, 
    organization_id: uuid.UUID, 
    skip: int = 0, 
    limit: int = 100,
    search: Optional[str] = None,
    is_active: Optional[bool] = None
) -> List[Tool]:
    """Get tools for an organization with optional filtering."""
    query = db.query(Tool).filter(Tool.organization_id == organization_id)
    
    if search:
        query = query.filter(
            or_(
                Tool.name.ilike(f"%{search}%"),
                Tool.description.ilike(f"%{search}%")
            )
        )
    
    if is_active is not None:
        query = query.filter(Tool.is_active == is_active)
    
    return query.order_by(desc(Tool.created_at)).offset(skip).limit(limit).all()


def update_tool(
    db: Session, 
    tool_id: uuid.UUID, 
    organization_id: uuid.UUID, 
    tool_data: ToolUpdate
) -> Optional[Tool]:
    """Update a tool."""
    tool = get_tool(db, tool_id, organization_id)
    if not tool:
        return None
    
    update_data = tool_data.dict(exclude_unset=True)
    
    # Handle headers conversion
    if "headers" in update_data and update_data["headers"] is not None:
        update_data["headers"] = [header.dict() for header in update_data["headers"]]
    
    # Handle parameters conversion
    if "parameters" in update_data and update_data["parameters"] is not None:
        update_data["parameters"] = [param.dict() for param in update_data["parameters"]]
    
    # Handle method uppercase
    if "method" in update_data:
        update_data["method"] = update_data["method"].upper()
    
    for field, value in update_data.items():
        setattr(tool, field, value)
    
    tool.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(tool)
    return tool


def delete_tool(db: Session, tool_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
    """Delete a tool."""
    tool = get_tool(db, tool_id, organization_id)
    if not tool:
        return False
    
    db.delete(tool)
    db.commit()
    return True


def add_tool_to_assistant(
    db: Session, 
    assistant_id: uuid.UUID, 
    tool_data: AssistantToolCreate,
    organization_id: uuid.UUID
) -> Optional[AssistantTool]:
    """Add a tool to an assistant."""
    # Verify tool belongs to organization
    tool = get_tool(db, tool_data.tool_id, organization_id)
    if not tool:
        return None
    
    # Check if assistant exists and belongs to organization
    assistant = db.query(Assistant).filter(
        and_(
            Assistant.id == assistant_id,
            Assistant.organization_id == organization_id
        )
    ).first()
    if not assistant:
        return None
    
    # Check if tool is already assigned to assistant
    existing = db.query(AssistantTool).filter(
        and_(
            AssistantTool.assistant_id == assistant_id,
            AssistantTool.tool_id == tool_data.tool_id
        )
    ).first()
    if existing:
        return existing
    
    assistant_tool = AssistantTool(
        assistant_id=assistant_id,
        tool_id=tool_data.tool_id,
        is_enabled=tool_data.is_enabled,
        priority=tool_data.priority
    )
    
    db.add(assistant_tool)
    db.commit()
    db.refresh(assistant_tool)
    return assistant_tool


def get_assistant_tools(
    db: Session, 
    assistant_id: uuid.UUID, 
    organization_id: uuid.UUID,
    enabled_only: bool = True
) -> List[AssistantTool]:
    """Get tools for an assistant."""
    query = db.query(AssistantTool).join(Tool).filter(
        and_(
            AssistantTool.assistant_id == assistant_id,
            Tool.organization_id == organization_id
        )
    )
    
    if enabled_only:
        query = query.filter(AssistantTool.is_enabled == True)
    
    return query.options(joinedload(AssistantTool.tool)).order_by(
        desc(AssistantTool.priority), 
        AssistantTool.created_at
    ).all()


def update_assistant_tool(
    db: Session, 
    assistant_tool_id: uuid.UUID, 
    organization_id: uuid.UUID, 
    tool_data: AssistantToolUpdate
) -> Optional[AssistantTool]:
    """Update assistant tool configuration."""
    assistant_tool = db.query(AssistantTool).join(Tool).filter(
        and_(
            AssistantTool.id == assistant_tool_id,
            Tool.organization_id == organization_id
        )
    ).first()
    
    if not assistant_tool:
        return None
    
    update_data = tool_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(assistant_tool, field, value)
    
    assistant_tool.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(assistant_tool)
    return assistant_tool


def remove_tool_from_assistant(
    db: Session, 
    assistant_tool_id: uuid.UUID, 
    organization_id: uuid.UUID
) -> bool:
    """Remove a tool from an assistant."""
    assistant_tool = db.query(AssistantTool).join(Tool).filter(
        and_(
            AssistantTool.id == assistant_tool_id,
            Tool.organization_id == organization_id
        )
    ).first()
    
    if not assistant_tool:
        return False
    
    db.delete(assistant_tool)
    db.commit()
    return True


def get_tool_by_name(db: Session, name: str, organization_id: uuid.UUID) -> Optional[Tool]:
    """Get a tool by name within an organization."""
    return db.query(Tool).filter(
        and_(
            Tool.name == name,
            Tool.organization_id == organization_id,
            Tool.is_active == True
        )
    ).first()


def get_available_tools_for_assistant(
    db: Session, 
    assistant_id: uuid.UUID, 
    organization_id: uuid.UUID
) -> List[Tool]:
    """Get all available tools that can be added to an assistant."""
    # Get tools already assigned to this assistant
    assigned_tool_ids = db.query(AssistantTool.tool_id).filter(
        AssistantTool.assistant_id == assistant_id
    ).subquery()
    
    # Get available tools (not assigned to this assistant)
    return db.query(Tool).filter(
        and_(
            Tool.organization_id == organization_id,
            Tool.is_active == True,
            ~Tool.id.in_(assigned_tool_ids)
        )
    ).order_by(Tool.name).all()
