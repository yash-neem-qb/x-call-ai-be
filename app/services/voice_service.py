"""
Voice service for fetching available voices from different providers.
Provides a unified interface for voice management across multiple TTS providers.
"""

import logging
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Voice:
    """Voice data structure."""
    id: str
    name: str
    provider: str
    gender: Optional[str] = None
    accent: Optional[str] = None
    language: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VoiceService:
    """Service for managing voices across different TTS providers."""
    
    def __init__(self):
        """Initialize the voice service."""
        self.elevenlabs_api_key = settings.elevenlabs_api_key
    
    async def get_voices(self, provider: str, page_size: int = 100) -> List[Voice]:
        """
        Get available voices for a specific provider.
        
        Args:
            provider: Voice provider (e.g., 'elevenlabs', 'openai', etc.)
            page_size: Number of voices to fetch (default: 100)
            
        Returns:
            List of Voice objects
            
        Raises:
            ValueError: If provider is not supported
            Exception: If API call fails
        """
        provider = provider.lower()
        
        if provider == 'elevenlabs' or provider == '11labs':
            return await self._get_elevenlabs_voices(page_size)
        elif provider == 'openai':
            return await self._get_openai_voices()
        else:
            raise ValueError(f"Unsupported voice provider: {provider}")
    
    async def _get_elevenlabs_voices(self, page_size: int = 100) -> List[Voice]:
        """
        Fetch voices from ElevenLabs API.
        
        Args:
            page_size: Number of voices to fetch
            
        Returns:
            List of ElevenLabs voices
        """
        try:
            url = "https://api.elevenlabs.io/v2/voices"
            headers = {
                "xi-api-key": self.elevenlabs_api_key
            }
            params = {
                "page_size": page_size
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ElevenLabs API error: {response.status} - {error_text}")
                        raise Exception(f"ElevenLabs API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    voices = data.get("voices", [])
                    
                    # Convert ElevenLabs format to our Voice format
                    voice_list = []
                    for voice_data in voices:
                        voice = Voice(
                            id=voice_data.get("voice_id", ""),
                            name=voice_data.get("name", ""),
                            provider="elevenlabs",
                            gender=voice_data.get("labels", {}).get("gender") or None,
                            accent=voice_data.get("labels", {}).get("accent") or None,
                            language=voice_data.get("labels", {}).get("language") or None,
                            description=voice_data.get("description") or None,
                            preview_url=voice_data.get("preview_url") or None,
                            metadata={
                                "category": voice_data.get("category"),
                                "use_case": voice_data.get("labels", {}).get("use_case"),
                                "labels": voice_data.get("labels", {}),
                                "settings": voice_data.get("settings", {}),
                                "sharing": voice_data.get("sharing", {}),
                                "high_quality_base_model_ids": voice_data.get("high_quality_base_model_ids", []),
                                "safety_control": voice_data.get("safety_control"),
                                "voice_verification": voice_data.get("voice_verification", {}),
                                "permission_on_resource": voice_data.get("permission_on_resource"),
                                "is_owner": voice_data.get("is_owner"),
                                "is_legacy": voice_data.get("is_legacy"),
                                "is_mixed": voice_data.get("is_mixed"),
                                "favorited_at_unix": voice_data.get("favorited_at_unix"),
                                "created_at_unix": voice_data.get("created_at_unix"),
                                "fine_tuning": voice_data.get("fine_tuning", {}),
                                "verified_languages": voice_data.get("verified_languages", []),
                                "samples": voice_data.get("samples"),
                                "available_for_tiers": voice_data.get("available_for_tiers", [])
                            }
                        )
                        voice_list.append(voice)
                    
                    logger.info(f"Fetched {len(voice_list)} voices from ElevenLabs")
                    return voice_list
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching ElevenLabs voices: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching ElevenLabs voices: {e}")
            raise
    
    async def _get_openai_voices(self) -> List[Voice]:
        """
        Fetch voices from OpenAI API.
        
        Returns:
            List of OpenAI voices
        """
        # OpenAI TTS voices are predefined
        openai_voices = [
            Voice(
                id="alloy",
                name="Alloy",
                provider="openai",
                gender="neutral",
                description="A neutral, balanced voice"
            ),
            Voice(
                id="echo",
                name="Echo",
                provider="openai",
                gender="male",
                description="A clear, confident voice"
            ),
            Voice(
                id="fable",
                name="Fable",
                provider="openai",
                gender="male",
                description="A warm, expressive voice"
            ),
            Voice(
                id="onyx",
                name="Onyx",
                provider="openai",
                gender="male",
                description="A deep, rich voice"
            ),
            Voice(
                id="nova",
                name="Nova",
                provider="openai",
                gender="female",
                description="A bright, energetic voice"
            ),
            Voice(
                id="shimmer",
                name="Shimmer",
                provider="openai",
                gender="female",
                description="A soft, gentle voice"
            )
        ]
        
        logger.info(f"Returned {len(openai_voices)} predefined OpenAI voices")
        return openai_voices
    
    def get_supported_providers(self) -> List[str]:
        """
        Get list of supported voice providers.
        
        Returns:
            List of supported provider names
        """
        return ["elevenlabs", "11labs", "openai"]


# Global voice service instance
voice_service = VoiceService()
