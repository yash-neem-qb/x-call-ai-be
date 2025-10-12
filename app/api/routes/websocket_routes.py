"""
WebSocket routes for media streaming.
Handles WebSocket connections for real-time audio processing.
"""

import json
import logging
import time
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session

from app.services.unified_pipeline import unified_pipeline_manager
from app.services.tts_service import elevenlabs_tts_service
from app.db.database import get_db
from app.db.crud import get_assistant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["websocket"])


@router.websocket('/websocket/media-stream')
async def handle_media_stream(websocket: WebSocket):
    """
    Handle WebSocket connections for media streaming.
    
    This endpoint manages the real-time audio streaming between Twilio and the unified pipeline.
    This is used by the create-call functionality and should NOT be changed.
    """
    logger.info("Client connected to media stream")
    await websocket.accept()
    
    # Track call information
    call_sid = None
    stream_sid = None
    pipeline = None
    
    try:
        # Ensure services are initialized
        await unified_pipeline_manager.initialize_services()
        
        async def send_audio_to_twilio(audio_data):
            """Send audio data to Twilio."""
            if not stream_sid:
                logger.warning("Cannot send audio - no stream SID")
                return

            try:
                audio_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_data
                    }
                }
                await websocket.send_json(audio_message)
            except Exception as e:
                logger.error(f"Error sending audio to Twilio: {e}")
        
        # Handle incoming messages from Twilio
        async for message in websocket.iter_text():
            data = json.loads(message)
            
            if data['event'] == 'start':
                # Call is starting
                stream_sid = data['start']['streamSid']
                call_sid = data['start'].get('callSid')
                
                logger.info(f"Media stream started: {stream_sid} for call: {call_sid}")
                
                # Get the pipeline for this call (should already be created by call creation)
                pipeline = await unified_pipeline_manager.get_pipeline(call_sid)
                if pipeline:
                    # Update the pipeline's audio callback to use the real WebSocket callback
                    pipeline.audio_callback = send_audio_to_twilio
                    
                    # Play greeting
                    await pipeline.play_greeting()
                else:
                    logger.warning(f"No pipeline found for call: {call_sid}")
                
            elif data['event'] == 'media' and stream_sid and pipeline:
                # Process incoming audio through unified pipeline
                audio_data = data['media']['payload']
                await pipeline.process_audio(audio_data)
                
            elif data['event'] == 'stop' and stream_sid:
                # Media stream is stopping - clear any pending audio
                logger.info(f"Media stream stopped: {stream_sid}")
                if pipeline:
                    # Cancel any ongoing TTS to clear audio buffers
                    try:
                        await elevenlabs_tts_service.cancel_speech(pipeline.tts_session_id)
                        logger.info(f"Audio buffers cleared for stream: {stream_sid}")
                    except Exception as e:
                        logger.warning(f"Error clearing audio buffers: {e}")
                # Don't end the call here - let the call status webhook handle it
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        # Clear audio buffers on disconnect
        if pipeline:
            try:
                await elevenlabs_tts_service.cancel_speech(pipeline.tts_session_id)
                logger.info("Audio buffers cleared on WebSocket disconnect")
            except Exception as e:
                logger.warning(f"Error clearing audio buffers on disconnect: {e}")
    except Exception as e:
        logger.error(f"Error in media stream handler: {e}")
    finally:
        logger.info("WebSocket connection closed")
        # Clear any remaining audio buffers
        if pipeline:
            try:
                await elevenlabs_tts_service.cancel_speech(pipeline.tts_session_id)
                logger.info("Final audio buffer cleanup completed")
            except Exception as e:
                logger.warning(f"Error in final audio cleanup: {e}")
        # Don't clean up pipeline here - let the call status webhook handle it


# Note: WebSocket-based assistant chat endpoint has been removed and replaced with WebRTC
# The new WebRTC endpoint is in webrtc_routes.py