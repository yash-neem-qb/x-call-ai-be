"""
API routes for assistant management.
Handles CRUD operations for assistants.
"""

import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Assistant
from app.models.assistant_schemas import (
    AssistantCreate,
    AssistantUpdate,
    AssistantResponse,
    AssistantList
)
from app.core.auth import require_write_permission, require_read_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/organizations", tags=["assistants"])


@router.post("/{organization_id}/assistants", response_model=AssistantResponse)
async def create_assistant(
    organization_id: uuid.UUID,
    assistant: AssistantCreate,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Create a new assistant for the organization.
    
    Args:
        organization_id: Organization ID
        assistant: Assistant data
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Created assistant
    """
    try:
        organization, membership = org_data
        
        # Create assistant
        system_prompt = assistant.model.systemPrompt
        if not system_prompt and assistant.model.messages:
            system_prompt = assistant.model.messages[0].content

        db_assistant = Assistant(
            name=assistant.name,
            organization_id=organization.id,
            team_id=assistant.team_id,
            voice_provider=assistant.voice.provider,
            voice_id=assistant.voice.voiceId,
            voice_model=assistant.voice.model,
            voice_stability=str(assistant.voice.stability),
            voice_similarity_boost=str(assistant.voice.similarityBoost),
            voice_speed=str(getattr(assistant.voice, "speed", 1.0)),
            llm_provider=assistant.model.provider,
            llm_model=assistant.model.model,
            llm_system_prompt=system_prompt,
            llm_max_tokens=assistant.model.maxTokens,
            llm_temperature=str(assistant.model.temperature) if assistant.model.temperature is not None else None,
            first_message=assistant.firstMessage,
            voicemail_message=assistant.voicemailMessage,
            end_call_message=assistant.endCallMessage,
            transcriber_provider=assistant.transcriber.provider,
            transcriber_model=assistant.transcriber.model,
            transcriber_language=assistant.transcriber.language,
            is_server_url_secret_set=assistant.isServerUrlSecretSet
        )
        
        db.add(db_assistant)
        db.commit()
        db.refresh(db_assistant)
        
        logger.info(f"Created assistant: {db_assistant.id} for organization: {organization.id}")
        return AssistantResponse(**db_assistant.to_dict())
        
    except Exception as e:
        logger.error(f"Error creating assistant: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")


@router.get("/{organization_id}/assistants", response_model=AssistantList)
async def get_assistants(
    organization_id: uuid.UUID,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of records to return"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get all assistants for the organization with pagination.
    
    Args:
        organization_id: Organization ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        List of assistants
    """
    try:
        organization, membership = org_data
        
        # Get assistants for the organization
        assistants = db.query(Assistant).filter(
            Assistant.organization_id == organization.id
        ).offset(skip).limit(limit).all()
        
        total = db.query(Assistant).filter(
            Assistant.organization_id == organization.id
        ).count()
        
        return AssistantList(
            items=[AssistantResponse(**assistant.to_dict()) for assistant in assistants],
            total=total,
            page=(skip // limit) + 1,
            pageSize=limit,
            totalPages=(total + limit - 1) // limit
        )
    except Exception as e:
        logger.error(f"Error getting assistants: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get assistants: {str(e)}")


@router.get("/{organization_id}/assistants/{assistant_id}", response_model=AssistantResponse)
async def get_assistant(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID = Path(..., description="Assistant ID"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get assistant by ID.
    
    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Assistant
    """
    try:
        organization, membership = org_data
        
        db_assistant = db.query(Assistant).filter(
            Assistant.id == assistant_id,
            Assistant.organization_id == organization.id
        ).first()
        
        if not db_assistant:
            raise HTTPException(status_code=404, detail=f"Assistant with ID {assistant_id} not found")
        
        return AssistantResponse(**db_assistant.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant {assistant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get assistant: {str(e)}")


@router.put("/{organization_id}/assistants/{assistant_id}", response_model=AssistantResponse)
async def update_assistant(
    organization_id: uuid.UUID,
    assistant: AssistantUpdate,
    assistant_id: uuid.UUID = Path(..., description="Assistant ID"),
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Update assistant.
    
    Args:
        organization_id: Organization ID
        assistant: Assistant data to update
        assistant_id: Assistant ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Updated assistant
    """
    try:
        organization, membership = org_data
        
        # Get the assistant
        db_assistant = db.query(Assistant).filter(
            Assistant.id == assistant_id,
            Assistant.organization_id == organization.id
        ).first()
        
        if not db_assistant:
            raise HTTPException(status_code=404, detail=f"Assistant with ID {assistant_id} not found")
        
        # Update individual fields based on the nested structure
        if assistant.name is not None:
            db_assistant.name = assistant.name
            
        # Update voice configuration
        if assistant.voice is not None:
            if assistant.voice.provider is not None:
                db_assistant.voice_provider = assistant.voice.provider
            if assistant.voice.voiceId is not None:
                db_assistant.voice_id = assistant.voice.voiceId
            if assistant.voice.model is not None:
                db_assistant.voice_model = assistant.voice.model
            if assistant.voice.stability is not None:
                db_assistant.voice_stability = str(assistant.voice.stability)
            if assistant.voice.similarityBoost is not None:
                db_assistant.voice_similarity_boost = str(assistant.voice.similarityBoost)
            if hasattr(assistant.voice, 'speed') and assistant.voice.speed is not None:
                db_assistant.voice_speed = str(assistant.voice.speed)
        
        # Update model configuration
        if assistant.model is not None:
            if assistant.model.provider is not None:
                db_assistant.llm_provider = assistant.model.provider
            if assistant.model.model is not None:
                db_assistant.llm_model = assistant.model.model
            if assistant.model.systemPrompt is not None:
                db_assistant.llm_system_prompt = assistant.model.systemPrompt
            elif assistant.model.messages and len(assistant.model.messages) > 0:
                # If systemPrompt is not provided but messages are, use the first message content
                db_assistant.llm_system_prompt = assistant.model.messages[0].content
            if assistant.model.maxTokens is not None:
                db_assistant.llm_max_tokens = assistant.model.maxTokens
            if assistant.model.temperature is not None:
                db_assistant.llm_temperature = str(assistant.model.temperature)
        
        # Update messages
        if assistant.firstMessage is not None:
            db_assistant.first_message = assistant.firstMessage
        if assistant.voicemailMessage is not None:
            db_assistant.voicemail_message = assistant.voicemailMessage
        if assistant.endCallMessage is not None:
            db_assistant.end_call_message = assistant.endCallMessage
            
        # Update transcriber configuration
        if assistant.transcriber is not None:
            if assistant.transcriber.provider is not None:
                db_assistant.transcriber_provider = assistant.transcriber.provider
            if assistant.transcriber.model is not None:
                db_assistant.transcriber_model = assistant.transcriber.model
            if assistant.transcriber.language is not None:
                db_assistant.transcriber_language = assistant.transcriber.language
                
        # Update server URL secret flag
        if assistant.isServerUrlSecretSet is not None:
            db_assistant.is_server_url_secret_set = assistant.isServerUrlSecretSet
        
        db.commit()
        db.refresh(db_assistant)
        
        logger.info(f"Updated assistant: {assistant_id}")
        return AssistantResponse(**db_assistant.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating assistant {assistant_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update assistant: {str(e)}")


@router.delete("/{organization_id}/assistants/{assistant_id}", response_model=dict)
async def delete_assistant(
    organization_id: uuid.UUID,
    assistant_id: uuid.UUID = Path(..., description="Assistant ID"),
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Delete assistant.
    
    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Success message
    """
    try:
        organization, membership = org_data
        
        # Get the assistant
        db_assistant = db.query(Assistant).filter(
            Assistant.id == assistant_id,
            Assistant.organization_id == organization.id
        ).first()
        
        if not db_assistant:
            raise HTTPException(status_code=404, detail=f"Assistant with ID {assistant_id} not found")
        
        db.delete(db_assistant)
        db.commit()
        
        logger.info(f"Deleted assistant: {assistant_id}")
        return {"success": True, "message": f"Assistant with ID {assistant_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting assistant {assistant_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete assistant: {str(e)}")
