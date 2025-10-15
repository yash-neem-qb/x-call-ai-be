"""
Campaign database models.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class CampaignStatus(PyEnum):
    """Campaign status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CampaignScheduleType(PyEnum):
    """Campaign schedule type enumeration."""
    NOW = "now"
    SCHEDULED = "scheduled"


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
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT, nullable=False)
    
    # Scheduling
    schedule_type = Column(SQLEnum(CampaignScheduleType), default=CampaignScheduleType.NOW, nullable=False)
    scheduled_at = Column(DateTime, nullable=True)
    
    # Campaign settings
    max_calls_per_hour = Column(Integer, default=50, nullable=False)
    retry_failed_calls = Column(Boolean, default=True, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    retry_delay_minutes = Column(Integer, default=30, nullable=False)
    
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
    status = Column(String(50), default="pending", nullable=False)  # pending, called, completed, failed, skipped
    call_attempts = Column(Integer, default=0, nullable=False)
    last_call_attempt = Column(DateTime, nullable=True)
    next_call_attempt = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="contacts")
