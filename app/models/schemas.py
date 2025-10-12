"""
Data models and schemas.
Defines Pydantic models for API request and response validation.
"""

import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class CreateCallRequest(BaseModel):
    """Request model for creating a call."""
    
    phone_number: str = Field(
        ...,
        description="Phone number to call in E.164 format (e.g., +1234567890)"
    )
    assistant_id: uuid.UUID = Field(
        ...,
        description="ID of the assistant to use for this call"
    )
    organization_id: uuid.UUID = Field(
        ...,
        description="ID of the organization making the call"
    )


class CreateCallResponse(BaseModel):
    """Response model for call creation."""
    
    success: bool = Field(
        ...,
        description="Whether the call was created successfully"
    )
    call_sid: str = Field(
        ...,
        description="The Twilio Call SID"
    )
    message: str = Field(
        ...,
        description="A message describing the result"
    )
    phone_number: str = Field(
        ...,
        description="The phone number that was called"
    )
    assistant_id: uuid.UUID = Field(
        ...,
        description="ID of the assistant used for this call"
    )
    organization_id: uuid.UUID = Field(
        ...,
        description="ID of the organization that made the call"
    )