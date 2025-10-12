"""
Configuration settings for the AI Voice Assistant application.
Centralizes all environment variables and configuration constants.
"""

import os
import re
from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Twilio Configuration
    twilio_account_sid: str
    twilio_auth_token: str
    phone_number_from: str
    
    # OpenAI Configuration
    openai_api_key: str
    # Default chat model for HTTP completions
    openai_chat_model: str = "gpt-4o"
    # Default realtime model for WS completions
    openai_realtime_model: str = "gpt-4o-realtime-preview"
    # Default max tokens for completions (name depends on model family)
    openai_max_tokens: int = 256
    # Default temperature for non-gpt-5 models
    openai_temperature: float = 0.8
    # Default system prompt for LLM
    openai_system_prompt: str = (
        "You are a helpful voice assistant. Be concise and conversational."
    )
    
    # ElevenLabs Configuration
    elevenlabs_api_key: str
    elevenlabs_voice_id: str = "hIssydxXZ1WuDorjx6Ic"
    
    # Deepgram Configuration
    deepgram_api_key: str = ""

    # Vector Database Configuration
    vector_qdrant_host: str = "localhost"
    vector_qdrant_port: int = 6333
    vector_qdrant_api_key: Optional[str] = None
    vector_qdrant_url: Optional[str] = None  # For cloud Qdrant
    
    # Collection Configuration
    vector_knowledge_collection_name: str = "knowledge_base"
    vector_size: int = 1536  # OpenAI embedding size
    vector_distance_metric: str = "Cosine"  # Cosine, Dot, or Euclid
    
    # Embedding Configuration
    vector_embedding_provider: str = "openai"  # openai or local
    vector_openai_embedding_model: str = "text-embedding-3-small"
    vector_local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Configuration
    vector_chunk_size: int = 1000
    vector_chunk_overlap: int = 200
    vector_max_tokens_per_chunk: int = 800
    
    # Search Configuration
    vector_default_search_limit: int = 10
    vector_similarity_threshold: float = 0.7
    
    # JWT Configuration
    jwt_secret_key: str = "your-secret-key-here"  # In production, use a secure secret key
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 10080  # 7 days (maximum practical expiration)
    
    # Server Configuration
    domain: str = ""
    port: int = 8000
    webhook_base_url: Optional[str] = None
    configure_webhooks: bool = False
    
    db_host: str = "metasetu.crogk2megxdl.ap-south-1.rds.amazonaws.com"
    db_user: str = "app"
    db_password: str = "metasetu321"
    db_name: str = "datasetu"
    database_url: str = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"
    sql_echo: bool = False

    log_level: str = "INFO"
    log_event_types: list = [
        'error', 'session.created', 'session.updated', 'input_audio_buffer.committed',
        'conversation.item.input_audio_transcription.delta', 
        'conversation.item.input_audio_transcription.completed'
    ]
    
    @validator('domain')
    def clean_domain(cls, v: str) -> str:
        """Clean domain by stripping protocols and trailing slashes."""
        return re.sub(r'(^\w+:|^)\/\/|\/+$', '', v)
    
    @validator('phone_number_from')
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        if not v.startswith('+'):
            raise ValueError('Phone number must start with +')
        if not v[1:].replace('-', '').replace(' ', '').replace('(', '').replace(')', '').isdigit():
            raise ValueError('Phone number must contain only digits after the +')
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """Validate that all required settings are present."""
    required_vars = [
        'twilio_account_sid', 'twilio_auth_token', 'phone_number_from',
        'openai_api_key', 'elevenlabs_api_key', 'deepgram_api_key'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(settings, var):
            missing_vars.append(var.upper())
    
    if missing_vars:
        raise ValueError(
            f'Missing required environment variables: {", ".join(missing_vars)}. '
            'Please set them in the .env file.'
        )

# Validate settings on import
validate_settings()
