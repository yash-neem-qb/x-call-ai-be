"""
Pydantic schemas for organization-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator
import uuid


class OrganizationBase(BaseModel):
    """Base organization schema."""
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    description: Optional[str] = Field(None, description="Organization description")
    settings: Optional[dict] = Field(default_factory=dict, description="Organization settings")


class OrganizationCreate(OrganizationBase):
    """Schema for creating a new organization."""
    slug: Optional[str] = Field(None, min_length=1, max_length=100, description="URL-friendly identifier")
    
    @validator('slug', pre=True, always=True)
    def generate_slug(cls, v, values):
        """Generate slug from name if not provided."""
        if not v and 'name' in values:
            # Simple slug generation - replace spaces with hyphens and lowercase
            return values['name'].lower().replace(' ', '-').replace('_', '-')
        return v


class OrganizationUpdate(BaseModel):
    """Schema for updating an organization."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    settings: Optional[dict] = None
    plan: Optional[str] = Field(None, pattern="^(free|pro|enterprise)$")
    max_users: Optional[int] = Field(None, ge=1)
    max_phone_numbers: Optional[int] = Field(None, ge=1)
    max_assistants: Optional[int] = Field(None, ge=1)
    is_active: Optional[bool] = None


class OrganizationResponse(OrganizationBase):
    """Schema for organization response."""
    id: uuid.UUID
    slug: str
    plan: str
    max_users: int
    max_phone_numbers: int
    max_assistants: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class OrganizationListResponse(BaseModel):
    """Schema for organization list response."""
    organizations: List[OrganizationResponse]
    total: int
    page: int
    per_page: int
    total_pages: int


class UserBase(BaseModel):
    """Base user schema."""
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$', description="User email address")
    first_name: Optional[str] = Field(None, max_length=100, description="User first name")
    last_name: Optional[str] = Field(None, max_length=100, description="User last name")
    preferences: Optional[dict] = Field(default_factory=dict, description="User preferences")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    organization_id: uuid.UUID = Field(..., description="Organization ID")
    role: str = Field("team", pattern="^(owner|admin|team)$", description="User role")


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, pattern="^(owner|admin|team)$")
    preferences: Optional[dict] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: uuid.UUID
    organization_id: uuid.UUID
    role: str
    is_active: bool
    is_verified: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TeamBase(BaseModel):
    """Base team schema."""
    name: str = Field(..., min_length=1, max_length=255, description="Team name")
    description: Optional[str] = Field(None, description="Team description")
    settings: Optional[dict] = Field(default_factory=dict, description="Team settings")


class TeamCreate(TeamBase):
    """Schema for creating a new team."""
    organization_id: uuid.UUID = Field(..., description="Organization ID")


class TeamUpdate(BaseModel):
    """Schema for updating a team."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class TeamResponse(TeamBase):
    """Schema for team response."""
    id: uuid.UUID
    organization_id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class TeamMemberCreate(BaseModel):
    """Schema for adding a user to a team."""
    user_id: uuid.UUID = Field(..., description="User ID")
    role: str = Field("team", pattern="^(owner|admin|team)$", description="Team role")


class TeamMemberResponse(BaseModel):
    """Schema for team member response."""
    id: uuid.UUID
    team_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    is_active: bool
    joined_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PhoneNumberBase(BaseModel):
    """Base phone number schema."""
    phone_number: str = Field(..., pattern=r'^\+[1-9]\d{1,14}$', description="Phone number in E.164 format")
    friendly_name: Optional[str] = Field(None, max_length=255, description="Friendly name for the phone number")
    settings: Optional[dict] = Field(default_factory=dict, description="Phone number settings")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant assigned to the phone number")


class PhoneNumberCreate(PhoneNumberBase):
    """Schema for creating a new phone number."""
    twilio_account_sid: str = Field(..., min_length=34, max_length=34, description="Twilio Account SID")
    twilio_auth_token: str = Field(..., min_length=32, description="Twilio Auth Token")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant assigned to the phone number")


class PhoneNumberUpdate(BaseModel):
    """Schema for updating a phone number."""
    friendly_name: Optional[str] = Field(None, max_length=255)
    settings: Optional[dict] = None
    is_active: Optional[bool] = None
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant assigned to the phone number")


class PhoneNumberResponse(PhoneNumberBase):
    """Schema for phone number response."""
    id: uuid.UUID
    organization_id: uuid.UUID
    twilio_account_sid: str
    twilio_phone_number_sid: Optional[str]
    voice_webhook_url: Optional[str]
    status_callback_url: Optional[str]
    is_active: bool
    is_configured: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CallBase(BaseModel):
    """Base call schema."""
    from_number: Optional[str] = Field(None, pattern=r'^\+[1-9]\d{1,14}$', description="Caller's phone number")
    to_number: Optional[str] = Field(None, pattern=r'^\+[1-9]\d{1,14}$', description="Called phone number")
    direction: str = Field(..., pattern="^(inbound|outbound|web)$", description="Call direction")
    call_data: Optional[dict] = Field(default_factory=dict, description="Call data and analytics")
    end_reason: Optional[str] = Field(None, pattern="^(user_hangup|assistant_hangup|system_error|timeout|network_error|call_completed|user_cancelled|assistant_cancelled)$", description="Reason why call ended")
    transcript_data: Optional[List[dict]] = Field(default_factory=list, description="JSON array of transcript messages")


class CallCreate(CallBase):
    """Schema for creating a new call."""
    organization_id: uuid.UUID = Field(..., description="Organization ID")
    phone_number_id: Optional[uuid.UUID] = Field(None, description="Phone number ID")
    assistant_id: Optional[uuid.UUID] = Field(None, description="Assistant ID")
    created_by: Optional[uuid.UUID] = Field(None, description="User who created the call")


class CallUpdate(BaseModel):
    """Schema for updating a call."""
    status: Optional[str] = Field(None, pattern="^(initiated|ringing|in_progress|completed|failed|busy|no_answer|cancelled)$")
    duration_seconds: Optional[int] = Field(None, ge=0)
    recording_url: Optional[str] = Field(None, max_length=500)
    transcription: Optional[str] = None
    call_data: Optional[dict] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    call_success: Optional[bool] = None
    call_summary: Optional[str] = None
    sentiment_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    analysis_completed: Optional[bool] = None
    detailed_analysis: Optional[dict] = None


class CallResponse(CallBase):
    """Schema for call response."""
    id: uuid.UUID
    twilio_call_sid: Optional[str]
    twilio_account_sid: Optional[str]
    organization_id: uuid.UUID
    phone_number_id: Optional[uuid.UUID]
    assistant_id: Optional[uuid.UUID]
    status: str
    duration_seconds: Optional[int]
    recording_url: Optional[str]
    transcription: Optional[str]
    quality_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    satisfaction_rating: Optional[int] = None
    call_success: Optional[bool] = None
    call_summary: Optional[str] = None
    analysis_completed: Optional[bool] = None
    detailed_analysis: Optional[dict] = None
    created_by: Optional[uuid.UUID]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CallListResponse(BaseModel):
    """Schema for call list response."""
    calls: List[CallResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
