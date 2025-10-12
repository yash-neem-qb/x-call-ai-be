"""
API routes for voice management.
Handles fetching available voices from different TTS providers.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.voice_service import voice_service, Voice

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["voices"])


class VoiceResponse(BaseModel):
    """Voice response model."""
    id: str
    name: str
    provider: str
    gender: Optional[str] = None
    accent: Optional[str] = None
    language: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True


class VoiceListResponse(BaseModel):
    """Voice list response model."""
    voices: List[VoiceResponse]
    provider: str
    total: int


@router.get("/voices", response_model=VoiceListResponse)
async def get_voices(
    provider: str = Query(..., description="Voice provider (e.g., 'elevenlabs', 'openai')"),
    page_size: int = Query(100, ge=1, le=500, description="Number of voices to fetch")
):
    """
    Get available voices for a specific provider.
    
    Args:
        provider: Voice provider (e.g., 'elevenlabs', 'openai')
        page_size: Number of voices to fetch (max 500)
        org_data: Organization and membership data
        
    Returns:
        List of available voices for the provider
        
    Example:
        GET /api/v1/voices?provider=elevenlabs&page_size=50
    """
    try:
        # Validate provider
        supported_providers = voice_service.get_supported_providers()
        if provider.lower() not in supported_providers:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported provider: {provider}. Supported providers: {', '.join(supported_providers)}"
            )
        
        # Fetch voices from the provider
        voices = await voice_service.get_voices(provider, page_size)
        
        # Convert to response format
        voice_responses = []
        for voice in voices:
            voice_responses.append(VoiceResponse(
                id=voice.id,
                name=voice.name,
                provider=voice.provider,
                gender=voice.gender or None,
                accent=voice.accent or None,
                language=voice.language or None,
                description=voice.description or None,
                preview_url=voice.preview_url or None,
                metadata=voice.metadata or {}
            ))
        
        logger.info(f"Retrieved {len(voice_responses)} voices for provider: {provider}")
        
        return VoiceListResponse(
            voices=voice_responses,
            provider=provider,
            total=len(voice_responses)
        )
        
    except ValueError as e:
        logger.error(f"Invalid provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching voices for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch voices: {str(e)}")


@router.get("/voices/providers")
async def get_supported_providers():
    """
    Get list of supported voice providers.
    
    Returns:
        List of supported voice providers
    """
    try:
        providers = voice_service.get_supported_providers()
        
        return {
            "providers": providers,
            "total": len(providers)
        }
        
    except Exception as e:
        logger.error(f"Error fetching supported providers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch supported providers: {str(e)}")
