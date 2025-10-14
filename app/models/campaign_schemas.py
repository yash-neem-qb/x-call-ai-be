"""
Pydantic schemas for campaign API endpoints.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class CampaignStatus(str, Enum):
    """Campaign status enumeration."""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class CampaignScheduleType(str, Enum):
    """Campaign schedule type enumeration."""
    NOW = "NOW"
    SCHEDULED = "SCHEDULED"


class CampaignContactStatus(str, Enum):
    """Campaign contact status enumeration."""
    PENDING = "pending"
    CALLED = "called"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Campaign Contact Schemas
class CampaignContactBase(BaseModel):
    """Base schema for campaign contact."""
    phone_number: str = Field(..., description="Contact phone number")
    name: Optional[str] = Field(None, description="Contact name")
    email: Optional[str] = Field(None, description="Contact email")

    @validator('phone_number')
    def validate_phone_number(cls, v):
        """Validate phone number format."""
        if not v or len(v) < 10:
            raise ValueError('Phone number must be at least 10 characters')
        return v

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class CampaignContactCreate(CampaignContactBase):
    """Schema for creating a campaign contact."""
    pass


class CampaignContactUpdate(BaseModel):
    """Schema for updating a campaign contact."""
    phone_number: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    status: Optional[CampaignContactStatus] = None
    call_attempts: Optional[int] = None
    last_call_attempt: Optional[datetime] = None
    next_call_attempt: Optional[datetime] = None


class CampaignContact(CampaignContactBase):
    """Schema for campaign contact response."""
    id: uuid.UUID
    campaign_id: uuid.UUID
    status: CampaignContactStatus
    call_attempts: int
    last_call_attempt: Optional[datetime] = None
    next_call_attempt: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Campaign Schemas
class CampaignBase(BaseModel):
    """Base schema for campaign."""
    name: str = Field(..., min_length=3, max_length=255, description="Campaign name")
    description: Optional[str] = Field(None, max_length=500, description="Campaign description")
    assistant_id: uuid.UUID = Field(..., description="Assistant ID")
    phone_number_id: uuid.UUID = Field(..., description="Phone number ID")
    
    # Scheduling
    schedule_type: CampaignScheduleType = Field(default=CampaignScheduleType.NOW, description="Schedule type")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled start time")


class CampaignCreate(CampaignBase):
    """Schema for creating a campaign."""
    contacts: Optional[List[CampaignContactCreate]] = Field(default=[], description="Campaign contacts from CSV")
    csv_data: Optional[List[Dict[str, Any]]] = Field(default=[], description="Raw CSV data")
    csv_headers: Optional[Dict[str, str]] = Field(default={}, description="CSV column mappings")


class CampaignUpdate(BaseModel):
    """Schema for updating a campaign."""
    name: Optional[str] = Field(None, min_length=3, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    assistant_id: Optional[uuid.UUID] = None
    phone_number_id: Optional[uuid.UUID] = None
    status: Optional[CampaignStatus] = None
    schedule_type: Optional[CampaignScheduleType] = None
    scheduled_at: Optional[datetime] = None


class Campaign(CampaignBase):
    """Schema for campaign response."""
    id: uuid.UUID
    organization_id: uuid.UUID
    status: CampaignStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    contacts: List[CampaignContact] = []

    class Config:
        from_attributes = True


# Campaign List Schemas
class CampaignListResponse(BaseModel):
    """Schema for campaign list response."""
    items: List[Campaign]
    total: int
    page: int
    page_size: int
    total_pages: int


# Campaign Action Schemas
class CampaignActionRequest(BaseModel):
    """Schema for campaign action requests."""
    action: str = Field(..., description="Action to perform (start, pause, stop, resume)")


class CampaignActionResponse(BaseModel):
    """Schema for campaign action response."""
    success: bool
    message: str
    campaign: Optional[Campaign] = None


# CSV Processing Schemas
class CSVUploadRequest(BaseModel):
    """Schema for CSV upload request."""
    csv_data: List[Dict[str, Any]] = Field(..., description="CSV data as list of dictionaries")
    headers: Dict[str, str] = Field(..., description="Column header mappings")
    campaign_id: Optional[uuid.UUID] = Field(None, description="Campaign ID if updating existing campaign")


class CSVUploadResponse(BaseModel):
    """Schema for CSV upload response."""
    success: bool
    message: str
    contacts_created: int
    contacts_updated: int
    errors: List[str] = []


# Campaign Analytics Schemas
class CampaignAnalytics(BaseModel):
    """Schema for campaign analytics."""
    campaign_id: uuid.UUID
    total_contacts: int
    total_calls: int
    completed_calls: int
    failed_calls: int
    success_rate: float
    average_duration: float
    total_duration: int
    total_cost: float
    calls_by_hour: List[Dict[str, Any]]
    calls_by_status: List[Dict[str, Any]]
    top_performing_assistants: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
