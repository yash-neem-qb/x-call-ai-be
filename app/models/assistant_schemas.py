"""
Pydantic schemas for assistant-related API operations.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    """Voice configuration schema."""
    
    model: str = Field(..., description="Voice model to use")
    voiceId: str = Field(..., description="Voice ID")
    provider: str = Field(..., description="Voice provider (e.g., '11labs')")
    stability: float = Field(0.5, description="Voice stability (0.0-1.0)")
    similarityBoost: float = Field(0.75, description="Voice similarity boost (0.0-1.0)")
    speed: float = Field(1.0, description="Voice speed multiplier")


class Message(BaseModel):
    """Message schema for system prompts."""
    
    role: str = Field(..., description="Message role (e.g., 'system', 'user', 'assistant')")
    content: str = Field(..., description="Message content")


class ModelConfig(BaseModel):
    """Model configuration schema."""
    
    model: str = Field(..., description="Model name (e.g., 'gpt-5-mini')")
    messages: List[Message] = Field(default_factory=list, description="System messages/prompts")
    provider: str = Field(..., description="Model provider (e.g., 'openai')")
    systemPrompt: Optional[str] = Field(None, description="System prompt text")
    maxTokens: Optional[int] = Field(None, description="Maximum tokens for responses")
    temperature: Optional[float] = Field(None, description="Sampling temperature")


class TranscriberConfig(BaseModel):
    """Transcriber configuration schema."""
    
    model: str = Field(..., description="Transcription model name")
    language: str = Field(..., description="Language code")
    provider: str = Field(..., description="Transcription provider")


class AssistantCreate(BaseModel):
    """Schema for creating a new assistant."""
    
    name: str = Field(..., description="Assistant name")
    team_id: Optional[uuid.UUID] = Field(None, description="Team ID (optional)")
    voice: VoiceConfig = Field(..., description="Voice configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    firstMessage: Optional[str] = Field(None, description="First message to say")
    voicemailMessage: Optional[str] = Field(None, description="Voicemail message")
    endCallMessage: Optional[str] = Field(None, description="End call message")
    transcriber: TranscriberConfig = Field(..., description="Transcriber configuration")
    isServerUrlSecretSet: Optional[bool] = Field(False, description="Whether server URL secret is set")


class AssistantUpdate(BaseModel):
    """Schema for updating an assistant."""
    
    name: Optional[str] = Field(None, description="Assistant name")
    voice: Optional[VoiceConfig] = Field(None, description="Voice configuration")
    model: Optional[ModelConfig] = Field(None, description="Model configuration")
    firstMessage: Optional[str] = Field(None, description="First message to say")
    voicemailMessage: Optional[str] = Field(None, description="Voicemail message")
    endCallMessage: Optional[str] = Field(None, description="End call message")
    transcriber: Optional[TranscriberConfig] = Field(None, description="Transcriber configuration")
    isServerUrlSecretSet: Optional[bool] = Field(None, description="Whether server URL secret is set")


class AssistantResponse(BaseModel):
    """Schema for assistant response."""
    
    id: uuid.UUID = Field(..., description="Assistant ID")
    name: str = Field(..., description="Assistant name")
    voice: VoiceConfig = Field(..., description="Voice configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    firstMessage: Optional[str] = Field(None, description="First message to say")
    voicemailMessage: Optional[str] = Field(None, description="Voicemail message")
    endCallMessage: Optional[str] = Field(None, description="End call message")
    transcriber: TranscriberConfig = Field(..., description="Transcriber configuration")
    isServerUrlSecretSet: bool = Field(..., description="Whether server URL secret is set")
    createdAt: datetime = Field(..., description="Creation timestamp")
    updatedAt: datetime = Field(..., description="Last update timestamp")


class AssistantList(BaseModel):
    """Schema for list of assistants."""
    
    items: List[AssistantResponse] = Field(..., description="List of assistants")
    total: int = Field(..., description="Total number of assistants")
    page: int = Field(1, description="Current page number")
    pageSize: int = Field(..., description="Page size")
    totalPages: int = Field(..., description="Total number of pages")
