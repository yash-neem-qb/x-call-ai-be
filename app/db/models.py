"""
SQLAlchemy models for database tables.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, Boolean, Text, Integer, ForeignKey, Enum, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.db.database import Base


class UserRole(enum.Enum):
    """User roles within an organization."""
    OWNER = "owner"
    ADMIN = "admin"
    TEAM = "team"


class CallStatus(enum.Enum):
    """Call status enumeration."""
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    CANCELLED = "cancelled"


class CallEndReason(enum.Enum):
    """Call end reason enumeration."""
    USER_HANGUP = "user_hangup"
    ASSISTANT_HANGUP = "assistant_hangup"
    SYSTEM_ERROR = "system_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    CALL_COMPLETED = "call_completed"
    USER_CANCELLED = "user_cancelled"
    ASSISTANT_CANCELLED = "assistant_cancelled"


class CallDirection(enum.Enum):
    """Call direction enumeration."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    WEB = "web"


class Assistant(Base):
    """Model for assistants table with normalized columns."""
    
    __tablename__ = "assistants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    
    # Organization and team relationships
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    
    # Voice configuration - normalized columns
    voice_provider = Column(String, nullable=False, default="11labs")  # 11labs, openai, etc.
    voice_id = Column(String, nullable=False)  # ElevenLabs voice ID
    voice_model = Column(String, nullable=False, default="eleven_flash_v2_5")  # TTS model
    voice_stability = Column(String, nullable=False, default="0.5")  # Voice stability
    voice_similarity_boost = Column(String, nullable=False, default="0.75")  # Voice similarity
    voice_speed = Column(String, nullable=False, default="1.0")  # Voice speed
    
    # LLM configuration - normalized columns
    llm_provider = Column(String, nullable=False, default="openai")  # openai, anthropic, etc.
    llm_model = Column(String, nullable=False, default="gpt-4.1-mini")  # LLM model name
    llm_system_prompt = Column(Text, nullable=True)  # System prompt for LLM
    llm_max_tokens = Column(Integer, nullable=True)  # Max tokens for LLM
    llm_temperature = Column(String, nullable=False, default="0.7")  # LLM temperature
    
    # Messages
    first_message = Column(String, nullable=True)
    voicemail_message = Column(String, nullable=True)
    end_call_message = Column(String, nullable=True)
    
    # Transcription configuration - normalized columns
    transcriber_provider = Column(String, nullable=False, default="deepgram")  # deepgram, openai, etc.
    transcriber_model = Column(String, nullable=False, default="nova-3")  # STT model
    transcriber_language = Column(String, nullable=False, default="en")  # Language code
    
    # Security
    is_server_url_secret_set = Column(Boolean, default=False)
    
    # RAG (Knowledge Base) configuration
    rag_enabled = Column(Boolean, default=True)
    rag_max_results = Column(Integer, default=3)
    rag_score_threshold = Column(String, default="0.7")
    rag_max_context_length = Column(Integer, default=2000)
    rag_config = Column(JSON, nullable=True, default=dict)  # Additional RAG settings
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="assistants")
    team = relationship("Team", back_populates="assistants")
    calls = relationship("Call", back_populates="assistant", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="assistant", cascade="all, delete-orphan")
    assigned_phone_numbers = relationship("PhoneNumber", back_populates="assistant")
    assistant_tools = relationship("AssistantTool", back_populates="assistant", cascade="all, delete-orphan")
    
    def __repr__(self):
        """String representation of the model."""
        return f"<Assistant(id='{self.id}', name='{self.name}', voice_id='{self.voice_id}')>"
    
    def to_dict(self):
        """Convert model to dictionary with normalized structure."""
        return {
            "id": str(self.id),
            "name": self.name,
            "organizationId": str(self.organization_id),
            "teamId": str(self.team_id) if self.team_id else None,
            # Voice configuration
            "voice": {
                "provider": self.voice_provider,
                "voiceId": self.voice_id,
                "model": self.voice_model,
                "stability": float(self.voice_stability),
                "similarityBoost": float(self.voice_similarity_boost),
                "speed": float(self.voice_speed)
            },
            # LLM configuration
            "model": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "systemPrompt": self.llm_system_prompt,
                "maxTokens": self.llm_max_tokens,
                "temperature": float(self.llm_temperature),
                "messages": [
                    {
                        "role": "system",
                        "content": self.llm_system_prompt or "You are a helpful voice assistant."
                    }
                ] if self.llm_system_prompt else []
            },
            # Messages
            "firstMessage": self.first_message,
            "voicemailMessage": self.voicemail_message,
            "endCallMessage": self.end_call_message,
            # Transcription configuration
            "transcriber": {
                "provider": self.transcriber_provider,
                "model": self.transcriber_model,
                "language": self.transcriber_language
            },
            # RAG (Knowledge Base) configuration
            "rag": {
                "enabled": self.rag_enabled,
                "maxResults": self.rag_max_results,
                "scoreThreshold": float(self.rag_score_threshold),
                "maxContextLength": self.rag_max_context_length,
                "config": self.rag_config or {}
            },
            "isServerUrlSecretSet": self.is_server_url_secret_set,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


class Organization(Base):
    """Model for organizations table."""
    
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)  # URL-friendly identifier
    description = Column(Text, nullable=True)
    
    # Organization settings
    settings = Column(JSON, nullable=True, default=dict)  # Flexible settings storage
    
    # Billing and limits
    plan = Column(String(50), nullable=False, default="free")  # free, pro, enterprise
    max_users = Column(Integer, nullable=False, default=5)
    max_phone_numbers = Column(Integer, nullable=False, default=1)
    max_assistants = Column(Integer, nullable=False, default=3)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="organization")
    members = relationship("OrganizationMember", back_populates="organization", cascade="all, delete-orphan")
    teams = relationship("Team", back_populates="organization", cascade="all, delete-orphan")
    phone_numbers = relationship("PhoneNumber", back_populates="organization", cascade="all, delete-orphan")
    assistants = relationship("Assistant", back_populates="organization", cascade="all, delete-orphan")
    calls = relationship("Call", back_populates="organization", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeBase", back_populates="organization", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Organization(id='{self.id}', name='{self.name}', slug='{self.slug}')>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "settings": self.settings or {},
            "plan": self.plan,
            "maxUsers": self.max_users,
            "maxPhoneNumbers": self.max_phone_numbers,
            "maxAssistants": self.max_assistants,
            "isActive": self.is_active,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


class User(Base):
    """Model for users table."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=True)  # For future auth implementation
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    
    # Organization relationship
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # User role
    role = Column(Enum(UserRole), nullable=False, default=UserRole.TEAM)
    
    # User preferences
    preferences = Column(JSON, nullable=True, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    organization_memberships = relationship("OrganizationMember", back_populates="user", cascade="all, delete-orphan")
    team_memberships = relationship("TeamMember", back_populates="user", cascade="all, delete-orphan")
    created_calls = relationship("Call", back_populates="created_by_user", foreign_keys="Call.created_by")
    
    def __repr__(self):
        return f"<User(id='{self.id}', email='{self.email}')>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "firstName": self.first_name,
            "lastName": self.last_name,
            "organizationId": str(self.organization_id) if self.organization_id else None,
            "role": self.role.value if self.role else None,
            "preferences": self.preferences or {},
            "isActive": self.is_active,
            "isVerified": self.is_verified,
            "lastLoginAt": self.last_login_at.isoformat() if self.last_login_at else None,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


class OrganizationMember(Base):
    """Model for organization memberships (many-to-many relationship)."""
    
    __tablename__ = "organization_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.TEAM)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="organization_memberships")
    organization = relationship("Organization", back_populates="members")
    
    def __repr__(self):
        return f"<OrganizationMember(user_id='{self.user_id}', organization_id='{self.organization_id}', role='{self.role.value}')>"


class Team(Base):
    """Model for teams within organizations."""
    
    __tablename__ = "teams"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Organization relationship
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Team settings
    settings = Column(JSON, nullable=True, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="teams")
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")
    assistants = relationship("Assistant", back_populates="team", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Team(id='{self.id}', name='{self.name}')>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "organizationId": str(self.organization_id),
            "settings": self.settings or {},
            "isActive": self.is_active,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


class TeamMember(Base):
    """Model for team memberships."""
    
    __tablename__ = "team_members"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.TEAM)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    joined_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="members")
    user = relationship("User", back_populates="team_memberships")
    
    def __repr__(self):
        return f"<TeamMember(team_id='{self.team_id}', user_id='{self.user_id}', role='{self.role.value}')>"


class PhoneNumber(Base):
    """Model for phone numbers with Twilio configuration."""
    
    __tablename__ = "phone_numbers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number = Column(String(20), nullable=False, unique=True, index=True)  # E.164 format
    
    # Organization relationship
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("assistants.id"), nullable=True)
    
    # Twilio configuration
    twilio_account_sid = Column(String(100), nullable=False)
    twilio_auth_token = Column(String(100), nullable=False)  # Should be encrypted in production
    twilio_phone_number_sid = Column(String(100), nullable=True)  # Twilio's internal ID
    
    # Webhook configuration
    voice_webhook_url = Column(String(500), nullable=True)
    status_callback_url = Column(String(500), nullable=True)
    
    # Phone number settings
    friendly_name = Column(String(255), nullable=True)
    settings = Column(JSON, nullable=True, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_configured = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="phone_numbers")
    assistant = relationship("Assistant", back_populates="assigned_phone_numbers")
    calls = relationship("Call", back_populates="phone_number", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PhoneNumber(id='{self.id}', phone_number='{self.phone_number}')>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "phoneNumber": self.phone_number,
            "organizationId": str(self.organization_id),
            "assistantId": str(self.assistant_id) if self.assistant_id else None,
            "twilioAccountSid": self.twilio_account_sid,
            "twilioPhoneNumberSid": self.twilio_phone_number_sid,
            "voiceWebhookUrl": self.voice_webhook_url,
            "statusCallbackUrl": self.status_callback_url,
            "friendlyName": self.friendly_name,
            "settings": self.settings or {},
            "isActive": self.is_active,
            "isConfigured": self.is_configured,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


class Call(Base):
    """Model for call records."""
    
    __tablename__ = "calls"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Twilio call information
    twilio_call_sid = Column(String(100), nullable=True, unique=True, index=True)
    twilio_account_sid = Column(String(100), nullable=True)
    
    # Organization and phone number relationships
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    phone_number_id = Column(UUID(as_uuid=True), ForeignKey("phone_numbers.id"), nullable=True)
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("assistants.id"), nullable=True)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=True)
    
    # Call details
    from_number = Column(String(20), nullable=True)  # Caller's number
    to_number = Column(String(20), nullable=True)    # Called number
    direction = Column(Enum(CallDirection), nullable=False)
    status = Column(Enum(CallStatus), nullable=False, default=CallStatus.INITIATED)
    
    # Web call specific fields
    session_id = Column(String(100), nullable=True)  # Web session ID
    
    # Call metadata
    duration_seconds = Column(Integer, nullable=True)
    recording_url = Column(String(500), nullable=True)
    transcription = Column(Text, nullable=True)
    end_reason = Column(Enum(CallEndReason), nullable=True)  # Reason why call ended
    
    # JSON transcript storage - stores array of messages with timestamps
    transcript_data = Column(JSON, nullable=True, default=lambda: [])  # [{"speaker": "user", "message": "Hello", "timestamp": "2024-01-01T10:00:00Z", "confidence": 0.95}]
    
    # Cost and billing information
    cost_usd = Column(Numeric(10, 4), nullable=True)  # Call cost in USD
    cost_currency = Column(String(3), nullable=True, default="USD")
    
    # Quality metrics
    quality_score = Column(Numeric(3, 2), nullable=True)  # 0.00 to 1.00
    sentiment_score = Column(Numeric(3, 2), nullable=True)  # -1.00 to 1.00
    satisfaction_rating = Column(Integer, nullable=True)  # 1-5 rating
    
    # Call analysis fields
    call_success = Column(Boolean, nullable=True)  # Whether the call achieved its goal
    call_summary = Column(Text, nullable=True)  # AI-generated summary of the call
    analysis_completed = Column(Boolean, default=False, nullable=False)  # Whether analysis has been completed
    detailed_analysis = Column(JSON, nullable=True, default=dict)  # Detailed analysis data for UI
    
    # Call data and analytics
    call_data = Column(JSON, nullable=True, default=dict)  # Store conversation data, analytics, etc.
    
    # User who created/initiated the call (for outbound calls)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="calls")
    phone_number = relationship("PhoneNumber", back_populates="calls")
    assistant = relationship("Assistant", back_populates="calls")
    campaign = relationship("Campaign", back_populates="calls")
    created_by_user = relationship("User", back_populates="created_calls", foreign_keys=[created_by])
    
    def __repr__(self):
        return f"<Call(id='{self.id}', twilio_call_sid='{self.twilio_call_sid}', status='{self.status.value}')>"
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "twilioCallSid": self.twilio_call_sid,
            "twilioAccountSid": self.twilio_account_sid,
            "organizationId": str(self.organization_id),
            "phoneNumberId": str(self.phone_number_id) if self.phone_number_id else None,
            "assistantId": str(self.assistant_id) if self.assistant_id else None,
            "fromNumber": self.from_number,
            "toNumber": self.to_number,
            "direction": self.direction.value,
            "status": self.status.value,
            "sessionId": self.session_id,
            "durationSeconds": self.duration_seconds,
            "recordingUrl": self.recording_url,
            "transcription": self.transcription,
            "transcriptData": self.transcript_data or [],
            "costUsd": float(self.cost_usd) if self.cost_usd else None,
            "costCurrency": self.cost_currency,
            "qualityScore": float(self.quality_score) if self.quality_score else None,
            "sentimentScore": float(self.sentiment_score) if self.sentiment_score else None,
            "satisfactionRating": self.satisfaction_rating,
            "callSuccess": self.call_success,
            "callSummary": self.call_summary,
            "analysisCompleted": self.analysis_completed,
            "detailedAnalysis": self.detailed_analysis or {},
            "callData": self.call_data or {},
            "createdBy": str(self.created_by) if self.created_by else None,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "endedAt": self.ended_at.isoformat() if self.ended_at else None,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat()
        }


class KnowledgeBase(Base):
    """Model for knowledge base documents."""
    
    __tablename__ = "knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("assistants.id"), nullable=True)
    
    # Document content
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=True)  # URL, file path, etc.
    
    # Metadata
    document_metadata = Column(JSON, nullable=True, default=dict)
    tags = Column(JSON, nullable=True, default=list)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organization = relationship("Organization", back_populates="knowledge_base")
    assistant = relationship("Assistant", back_populates="knowledge_base")
    
    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', title='{self.title}', organization_id='{self.organization_id}')>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "assistant_id": self.assistant_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "metadata": self.document_metadata if isinstance(self.document_metadata, dict) else {},
            "tags": self.tags or [],
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class CampaignStatus(enum.Enum):
    """Campaign status enumeration."""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class CampaignScheduleType(enum.Enum):
    """Campaign schedule type enumeration."""
    NOW = "NOW"
    SCHEDULED = "SCHEDULED"


class CampaignContactStatus(enum.Enum):
    """Campaign contact status enumeration."""
    PENDING = "pending"
    CALLED = "called"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Campaign(Base):
    """Campaign model for storing campaign information."""
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    assistant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    phone_number_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Campaign details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(CampaignStatus), default=CampaignStatus.DRAFT, nullable=False)
    
    # Scheduling
    schedule_type = Column(Enum(CampaignScheduleType), default=CampaignScheduleType.NOW, nullable=False)
    scheduled_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    contacts = relationship("CampaignContact", back_populates="campaign", cascade="all, delete-orphan")
    calls = relationship("Call", back_populates="campaign")


class CampaignContact(Base):
    """Campaign contact model for storing individual contacts from CSV."""
    __tablename__ = "campaign_contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False, index=True)
    
    # Contact information
    phone_number = Column(String(20), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    
    # Contact status
    status = Column(Enum(CampaignContactStatus), default=CampaignContactStatus.PENDING, nullable=False)
    call_attempts = Column(Integer, default=0, nullable=False)
    last_call_attempt = Column(DateTime, nullable=True)
    next_call_attempt = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="contacts")


class Tool(Base):
    """Tool model for external function calling."""
    __tablename__ = "tools"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # HTTP configuration
    method = Column(String(10), nullable=False)  # GET, POST, PUT, DELETE, etc.
    url = Column(Text, nullable=False)
    headers = Column(JSON, nullable=True)  # Custom headers
    parameters = Column(JSON, nullable=True)  # URL parameters
    body_schema = Column(JSON, nullable=True)  # Request body schema for POST/PUT
    
    # Organization and assistant relationships
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Tool configuration
    timeout_seconds = Column(Integer, default=30, nullable=False)
    retry_count = Column(Integer, default=3, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    organization = relationship("Organization", backref="tools")
    assistant_tools = relationship("AssistantTool", back_populates="tool", cascade="all, delete-orphan")


class AssistantTool(Base):
    """Many-to-many relationship between assistants and tools."""
    __tablename__ = "assistant_tools"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Foreign keys
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("assistants.id"), nullable=False)
    tool_id = Column(UUID(as_uuid=True), ForeignKey("tools.id"), nullable=False)
    
    # Tool configuration for this assistant
    is_enabled = Column(Boolean, default=True, nullable=False)
    priority = Column(Integer, default=0, nullable=False)  # Higher number = higher priority
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    assistant = relationship("Assistant", back_populates="assistant_tools")
    tool = relationship("Tool", back_populates="assistant_tools")