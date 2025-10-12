"""
WebRTC signaling routes for assistant communication.
Handles WebRTC peer connection signaling and audio streaming.
"""

import json
import logging
import uuid
import asyncio
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session

from app.services.unified_pipeline import unified_pipeline_manager
from app.services.async_call_service import async_call_service
from app.db.database import get_db
from app.db.crud import get_assistant, create_call_log, update_call_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["webrtc"])


@router.websocket('/webrtc/signaling/{assistant_id}')
async def handle_webrtc_signaling(
    websocket: WebSocket, 
    assistant_id: str,
    organization_id: str = Query(..., description="Organization ID"),
    token: str = Query(None, description="JWT Token for authentication")
):
    """
    Handle WebRTC signaling for direct assistant communication.
    
    This endpoint manages WebRTC peer connection signaling and audio streaming
    between the frontend and the unified pipeline, enabling full-duplex audio.
    
    Args:
        websocket: WebSocket connection for signaling
        assistant_id: Assistant ID to communicate with
        organization_id: Organization ID (query parameter)
    """
    logger.info(f"WebRTC signaling connection - Assistant: {assistant_id}, Organization: {organization_id}")
    
    # Track session information
    session_id = f"webrtc_{uuid.uuid4()}"
    pipeline = None
    peer_connection_ready = False
    call_log_id = None
    
    try:
        # Accept the WebSocket connection
        await websocket.accept()
        
        # Verify assistant exists and get configuration
        db = next(get_db())
        try:
            assistant = get_assistant(db, uuid.UUID(assistant_id))
            if not assistant:
                await websocket.send_json({
                    "type": "error",
                    "message": "Assistant not found"
                })
                await websocket.close(code=4004, reason="Assistant not found")
                return
            
            # Verify organization access
            if str(assistant.organization_id) != organization_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "Access denied - assistant belongs to different organization"
                })
                await websocket.close(code=4003, reason="Access denied")
                return
                
        finally:
            db.close()
        
        async def send_audio_to_client(audio_data: str, format: str = "pcm_16000"):
            """Send audio data to WebRTC client."""
            try:
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_data,
                    "format": format
                })
            except Exception as e:
                logger.error(f"Error sending audio to WebRTC client: {e}")

        async def send_text_to_client(message_type: str, content: str, is_final: bool = False):
            """Send text message to WebRTC client."""
            try:
                await websocket.send_json({
                    "type": "text",
                    "messageType": message_type,
                    "content": content,
                    "isFinal": is_final
                })
            except Exception as e:
                logger.error(f"Error sending text to WebRTC client: {e}")
        
        # Create unified pipeline for this WebRTC session
        try:
            # Configure pipeline for WebRTC client (use PCM format for better quality)
            pipeline_config = {
                "tts_output_format": "pcm_16000",  # High-quality format for WebRTC
                "stt_model": "nova-2",  # Use stable nova-2 model that works with our API key
                "llm_model": "gpt-4o-mini"  # Fast response model
            }
            
            pipeline = await unified_pipeline_manager.create_pipeline(
                call_sid=session_id,
                assistant_id=uuid.UUID(assistant_id),
                audio_callback=send_audio_to_client,
                text_callback=send_text_to_client,
                config_overrides=pipeline_config
            )
            logger.info(f"Unified pipeline created for WebRTC session: {session_id}")
            
            # Log the call start
            try:
                call_data = {
                    "organizationId": uuid.UUID(organization_id),
                    "assistantId": uuid.UUID(assistant_id),
                    "direction": "web",
                    "status": "initiated",
                    "sessionId": session_id,
                    "twilioCallSid": session_id,  # Use session_id as the call identifier for web calls
                    "startedAt": datetime.utcnow()
                }
                db = next(get_db())
                try:
                    call_log = create_call_log(db, call_data)
                    call_log_id = call_log.id
                    logger.info(f"🔍 WebRTC call logged with ID: {call_log_id}, Session ID: {session_id}, twilio_call_sid: {call_log.twilio_call_sid}")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Failed to log WebRTC call: {e}")
            
            # Connection established - no initial greeting needed
            
        except ValueError as e:
            logger.error(f"Failed to create pipeline: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to initialize assistant: {str(e)}"
            })
            await websocket.close(code=4000, reason="Pipeline initialization failed")
            return
        
        # Handle WebRTC signaling messages (simplified)
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'ready':
                    # Client is ready - no WebRTC handshake needed
                    logger.info("Client ready for audio communication")
                    peer_connection_ready = True
                    
                    # Send a test greeting to keep the pipeline active
                    if pipeline:
                        await asyncio.sleep(0.5)  # Wait a bit for connection to stabilize
                        await pipeline.play_greeting()
                    
                elif message_type == 'audio' and peer_connection_ready and pipeline:
                    # Process audio data through the unified pipeline
                    audio_data = data.get('data')
                    if audio_data:
                        # Process audio data through the unified pipeline
                        import base64
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            # Only log audio processing errors, not every chunk
                        except Exception as e:
                            logger.error(f"Error decoding audio data: {e}")
                        
                        await pipeline.process_audio(audio_data)
                    else:
                        logger.warning("Received audio data but no data payload")
                
                elif message_type == 'text' and pipeline:
                    # Handle text control messages (start/stop signals)
                    text_content = data.get('content', "")
                    if text_content:
                        logger.info(f"Received text control message: {text_content}")
                        
                        if text_content == 'start':
                            # Client is starting to send audio
                            logger.info("Client started audio input")
                            # Could set pipeline state to listening here if needed
                            
                        elif text_content == 'stop':
                            # Client finished sending audio
                            logger.info("Client stopped audio input")
                            # Could trigger final processing here if needed
                            
                        else:
                            # This is actual text content for the LLM
                            logger.info(f"Processing text input: {text_content}")
                            # Add to conversation history and generate response
                            await pipeline._handle_stt_complete(pipeline.stt_session_id, text_content)
                    else:
                        logger.warning("Received text message but no content")
                
                elif message_type == 'ping':
                    # Respond to ping with pong
                    await websocket.send_json({"type": "pong"})
                
                elif message_type == 'disconnect':
                    # Client is disconnecting
                    logger.info("Client requested disconnect")
                    break
                
                else:
                    logger.warning(f"Unknown WebRTC message type: {message_type}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in WebRTC message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message"
                })
            except Exception as e:
                logger.error(f"Error processing WebRTC message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info("WebRTC signaling WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebRTC signaling handler: {e}")
    finally:
        logger.info("WebRTC signaling connection closed")
        
        # Log the call end asynchronously
        if session_id:
            try:
                # Determine end reason based on how the call ended
                end_reason = "user_hangup"  # Default for web calls (user closed browser/tab)
                
                # Calculate duration if we have the call log
                duration_seconds = None
                if call_log_id:
                    try:
                        db = next(get_db())
                        try:
                            from app.db.base_crud import get_call_log
                            call_log = get_call_log(db, call_log_id)
                            if call_log and call_log.started_at:
                                duration_seconds = int((datetime.utcnow() - call_log.started_at).total_seconds())
                        finally:
                            db.close()
                    except Exception as e:
                        logger.warning(f"Could not calculate call duration: {e}")
                
                # Update call end status asynchronously using the async service
                await async_call_service.end_call(
                    twilio_call_sid=session_id,  # Use session_id as the identifier
                    end_reason=end_reason,
                    duration_seconds=duration_seconds
                )
                logger.info(f"WebRTC call end queued for update: {session_id}")
            except Exception as e:
                logger.error(f"Failed to queue WebRTC call end update: {e}")
        
        # Clean up the pipeline
        if pipeline:
            await unified_pipeline_manager.remove_pipeline(session_id)


# SDP generation function removed - using simplified WebSocket-based approach


@router.get('/webrtc/status')
async def get_webrtc_status():
    """
    Get WebRTC service status and capabilities.
    
    Returns:
        dict: WebRTC service status information
    """
    return {
        "status": "active",
        "capabilities": {
            "audio_codecs": ["opus", "pcmu", "pcma"],
            "sample_rates": [8000, 16000, 48000],
            "echo_cancellation": True,
            "noise_suppression": True,
            "auto_gain_control": True
        },
        "ice_servers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"}
        ]
    }
