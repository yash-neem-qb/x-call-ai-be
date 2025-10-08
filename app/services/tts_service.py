"""
Text-to-Speech service using ElevenLabs.
Provides a clean interface for converting text to speech.
"""

import json
import logging
import asyncio
import aiohttp
import websockets
import base64
from typing import Optional, Dict, Any, Callable, Coroutine

from app.config.settings import settings

logger = logging.getLogger(__name__)


class ElevenLabsTTSService:
    """Service for Text-to-Speech operations using ElevenLabs API."""
    
    def __init__(self):
        """Initialize the TTS service."""
        self.api_key = settings.elevenlabs_api_key
        self.voice_id = settings.elevenlabs_voice_id
        self.model_id = "eleven_flash_v2_5"  # Fastest model for lowest latency
        self.default_output_format = "ulaw_8000"  # Default format compatible with Twilio
        self.active_connections = {}
        self._greeting_audio = None
    
    async def initialize(self):
        """
        Initialize the TTS service.
        """
        try:
            logger.info("TTS service initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing TTS service: {e}")
            return False
    
    # Removed unused greeting preloading to reduce complexity and dependencies.
    
    async def create_tts_session(self, session_id: str, 
                                on_audio_chunk: Callable[[str, str], Coroutine],
                                voice_id: str = None,
                                model_id: str = None,
                                stability: float = None,
                                similarity_boost: float = None,
                                output_format: str = None,
                                on_completion: Optional[Callable[[str], Coroutine]] = None) -> bool:
        """
        Create a new TTS session.
        
        Args:
            session_id: Unique identifier for this session
            on_audio_chunk: Callback for audio chunks
            voice_id: Voice ID to use (optional)
            model_id: Model ID to use (optional)
            stability: Voice stability setting (optional)
            similarity_boost: Voice similarity boost setting (optional)
            output_format: Audio output format (optional)
            on_completion: Completion callback (optional)
            
        Returns:
            bool: True if session was created successfully
        """
        try:
            # Use provided parameters or fall back to defaults
            used_voice_id = voice_id if voice_id is not None else self.voice_id
            used_model_id = model_id if model_id is not None else self.model_id
            used_stability = stability if stability is not None else 0.5
            used_similarity_boost = similarity_boost if similarity_boost is not None else 0.8
            used_output_format = output_format if output_format is not None else self.default_output_format
            
            logger.info(f"Creating TTS session with voice: {used_voice_id}, model: {used_model_id}, format: {used_output_format}")
            
            # Connect to ElevenLabs WebSocket API with fastest settings and maximum timeout
            elevenlabs_url = (
                f"wss://api.elevenlabs.io/v1/text-to-speech/{used_voice_id}/stream-input"
                f"?model_id={used_model_id}&output_format={used_output_format}&inactivity_timeout=180"
            )
            
            elevenlabs_ws = await websockets.connect(
                elevenlabs_url,
                extra_headers={
                    "xi-api-key": self.api_key
                },
                ping_interval=8000,  # 5 minutes - stable connection
                ping_timeout=8000     # 30 seconds - tolerant timeout
            )
            
            # Initialize the connection with fastest generation config
            init_message = {
                "text": " ",
                "voice_settings": {
                    "stability": used_stability,
                    "similarity_boost": used_similarity_boost,
                    "use_speaker_boost": False,
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 80, 120, 150]  # Faster chunks for lower latency
                }
            }
            await elevenlabs_ws.send(json.dumps(init_message))
            
            # Store connection and callback with voice settings
            self.active_connections[session_id] = {
                "websocket": elevenlabs_ws,
                "on_audio_chunk": on_audio_chunk,
                "on_completion": on_completion,
                "voice_id": used_voice_id,
                "model_id": used_model_id,
                "stability": used_stability,
                "similarity_boost": used_similarity_boost,
                "output_format": used_output_format,
                "task": None,
                "last_activity": asyncio.get_event_loop().time(),
                # When True, drop outgoing audio chunks (used for barge-in)
                "mute_output": False
            }
            
            # Start listening for audio chunks
            self.active_connections[session_id]["task"] = asyncio.create_task(
                self._handle_audio_chunks(session_id)
            )
            
            logger.info(f"TTS session created: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating TTS session: {e}")
            return False
    
    async def _handle_audio_chunks(self, session_id: str):
        """
        Handle audio chunks from ElevenLabs with optimized speed.
        
        Args:
            session_id: The session identifier
        """
        if session_id not in self.active_connections:
            logger.error(f"No active connection for session: {session_id}")
            return
        
        connection = self.active_connections[session_id]
        elevenlabs_ws = connection["websocket"]
        on_audio_chunk = connection["on_audio_chunk"]
        
        try:
            while True:
                try:
                    # Wait for audio message with timeout
                    message = await asyncio.wait_for(elevenlabs_ws.recv(), timeout=300.0)
                    data = json.loads(message)
                    
                    # Update last activity time
                    connection["last_activity"] = asyncio.get_event_loop().time()
                    
                    if data.get("audio"):
                        # Print time-only when ElevenLabs generates first audio chunk per generation
                        if not connection.get("_printed_first_chunk_time"):
                            from datetime import datetime
                            print(datetime.now().strftime("%H:%M:%S.%f")[:-3])
                            connection["_printed_first_chunk_time"] = True
                        # Send audio to callback unless muted (barge-in)
                        if not connection.get("mute_output", False):
                            # Process audio chunk
                            await on_audio_chunk(session_id, data["audio"])
                        else:
                            logger.debug(f"Dropping TTS audio chunk for muted session: {session_id}")
                    
                    elif data.get("isFinal"):
                        # Don't close the connection - just log completion
                        logger.info(f"TTS stream completed for session: {session_id}")
                        # Reset marker for next generation
                        if connection.get("_printed_first_chunk_time"):
                            connection["_printed_first_chunk_time"] = False
                        
                        # Notify completion callback if provided
                        on_completion = connection.get("on_completion")
                        if on_completion:
                            try:
                                await on_completion(session_id)
                            except Exception as e:
                                logger.error(f"Error in TTS completion callback: {e}")
                        
                except asyncio.TimeoutError:
                    # No audio available yet, continue polling
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"TTS connection closed unexpectedly: {session_id}")
                    # Don't auto-reconnect - let the system handle it naturally
                    break
                    
        except Exception as e:
            logger.error(f"Error in audio handler: {e}")
        finally:
            # Don't remove the connection here - it might be reused
            pass
    
    async def speak_text(self, session_id: str, text: str, is_final: bool = False) -> bool:
        """
        Convert text to speech.
        
        Args:
            session_id: The session identifier
            text: The text to convert to speech
            is_final: Whether this is the final chunk
            
        Returns:
            bool: True if text was processed successfully
        """
        logger.info(f"🎤 TTS speak_text called for session {session_id}: '{text[:50]}...' (is_final: {is_final})")
        # Create session if it doesn't exist
        if session_id not in self.active_connections:
            logger.error(f"No active connection for session: {session_id}")
            return False
        
        try:
            connection = self.active_connections[session_id]
            elevenlabs_ws = connection["websocket"]
            
            # Get stored voice settings
            voice_id = connection.get("voice_id")
            model_id = connection.get("model_id")
            stability = connection.get("stability", 0.5)
            similarity_boost = connection.get("similarity_boost", 0.8)
            
            # Update last activity time
            connection["last_activity"] = asyncio.get_event_loop().time()
            
            if is_final:
                # Send empty text to indicate end of stream
                await elevenlabs_ws.send(json.dumps({"text": "", "flush": True}))
                logger.debug(f"Sent final chunk signal for session: {session_id}")
            else:
                # Send the text chunk with fastest generation trigger
                text_message = {
                    "text": text,
                    "try_trigger_generation": True,  # Force immediate generation
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "use_speaker_boost": False,
                    }
                }
                await elevenlabs_ws.send(json.dumps(text_message))
                logger.info(f"🎤 Sent text to ElevenLabs: '{text[:20]}...' ({len(text)} chars) with fast generation")
            
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection closed during text processing: {session_id}")
            # Connection lost - return False to let caller handle reconnection
            return False
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return False

    async def cancel_speech(self, session_id: str) -> bool:
        """
        Immediately stop current speech output and mute further chunks.
        Sends a flush signal to ElevenLabs and mutes callback delivery.
        """
        if session_id not in self.active_connections:
            logger.warning(f"No active connection to cancel for session: {session_id}")
            return True
        try:
            connection = self.active_connections[session_id]
            connection["mute_output"] = True
            elevenlabs_ws = connection["websocket"]
            if elevenlabs_ws and elevenlabs_ws.open:
                try:
                    await elevenlabs_ws.send(json.dumps({"text": "", "flush": True}))
                except Exception:
                    # Best-effort flush
                    pass
            logger.info(f"TTS cancelled and muted for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling TTS speech: {e}")
            return False

    # Removed unused manual mute setter to simplify the interface.
    
    # Removed unused cached greeting sender to streamline TTS service.
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close a TTS session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            bool: True if session was closed successfully
        """
        if session_id not in self.active_connections:
            logger.warning(f"No active connection to close for session: {session_id}")
            return True
        
        try:
            connection = self.active_connections[session_id]
            
            # Cancel the task if running
            if connection["task"] and not connection["task"].done():
                connection["task"].cancel()
                try:
                    await connection["task"]
                except asyncio.CancelledError:
                    pass
            
            # Close the WebSocket
            if connection["websocket"].open:
                await connection["websocket"].close()
            
            # Remove the connection
            self.active_connections.pop(session_id, None)
            
            logger.info(f"TTS session closed: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing TTS session: {e}")
            return False


# Global ElevenLabs TTS service instance
elevenlabs_tts_service = ElevenLabsTTSService()
