"""
Pydantic schemas for call logs models.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class CallLogBase(BaseModel):
    """Base call log schema."""
    organization_id: uuid.UUID
    phone_number_id: Optional[uuid.UUID] = None
    assistant_id: Optional[uuid.UUID] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    direction: str = Field(..., description="Call direction: inbound, outbound, or web")
    session_id: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    cost_usd: Optional[float] = None
    created_by: Optional[uuid.UUID] = None


class CallLogCreate(CallLogBase):
    """Schema for creating a call log."""
    pass


class CallLogUpdate(BaseModel):
    """Schema for updating a call log."""
    status: Optional[str] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    cost_usd: Optional[float] = None
    transcription: Optional[str] = None
    transcript_data: Optional[list] = None
    recording_url: Optional[str] = None
    call_success: Optional[bool] = None
    call_summary: Optional[str] = None
    sentiment_score: Optional[float] = None
    analysis_completed: Optional[bool] = None
    detailed_analysis: Optional[dict] = None


class CallLogResponse(CallLogBase):
    """Schema for call log response."""
    id: uuid.UUID
    twilio_call_sid: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    status: str
    recording_url: Optional[str] = None
    transcription: Optional[str] = None
    transcript_data: Optional[list] = None
    cost_currency: Optional[str] = None
    quality_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    satisfaction_rating: Optional[int] = None
    call_success: Optional[bool] = None
    call_summary: Optional[str] = None
    analysis_completed: Optional[bool] = None
    detailed_analysis: Optional[dict] = None
    call_data: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
