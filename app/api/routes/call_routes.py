"""
API routes for call management.
Handles HTTP endpoints for creating and managing calls.
"""

import logging
from fastapi import APIRouter, HTTPException, Request, Response, Form
from fastapi.responses import JSONResponse, PlainTextResponse

from app.models.schemas import CreateCallRequest, CreateCallResponse
from app.services.twilio_service import twilio_service
from app.services.unified_pipeline import unified_pipeline_manager
from app.db.database import get_db
from app.db.crud import create_call_log, update_call_log
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["calls"])


@router.get('/', response_class=JSONResponse)
async def index_page():
    """Root endpoint providing API information."""
    return {
        "message": "Voice Assistant API is running!",
        "version": "1.0.0",
        "endpoints": {
            "create_call": "/api/v1/create-call",
            "call_status": "/api/v1/call-status",
            "websocket": "/media-stream"
        }
    }


@router.post('/create-call', response_model=CreateCallResponse)
async def create_call(request: CreateCallRequest):
    """
    Create a new AI-powered voice call to the specified phone number.
    
    This endpoint initiates a call using Twilio and connects it to the unified pipeline.
    
    Args:
        request: The call creation request containing the phone number, assistant ID, and organization ID
        
    Returns:
        CreateCallResponse: The call creation response with call details
    """
    try:
        logger.info(f"Creating call to {request.phone_number} with assistant {request.assistant_id}")
        
        # Initialize unified pipeline services if needed
        await unified_pipeline_manager.initialize_services()
        
        # Make the call first to get the call SID
        call_sid = await twilio_service.make_call(
            request.phone_number,
            status_callback_url=f"/api/v1/call-status"
        )
        
        # Log the outbound call
        try:
            call_data = {
                "organizationId": request.organization_id,
                "assistantId": request.assistant_id,
                "direction": "outbound",
                "status": "initiated",
                "fromNumber": None,  # We don't have our number in the request
                "toNumber": request.phone_number,
                "startedAt": datetime.utcnow(),
                "twilioCallSid": call_sid  # Store the Twilio Call SID
            }
            db = next(get_db())
            try:
                call_log = create_call_log(db, call_data)
                logger.info(f"Outbound call logged with ID: {call_log.id}, Twilio SID: {call_sid}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to log outbound call: {e}")
        
        # Create a dummy audio callback for pipeline initialization
        # The real callback will be set up in the WebSocket handler
        async def dummy_audio_callback(audio_data: str):
            logger.debug(f"Dummy audio callback called for call {call_sid}")
        
        # Create unified pipeline for this call
        try:
            pipeline = await unified_pipeline_manager.create_pipeline(
                call_sid=call_sid,
                assistant_id=request.assistant_id,
                audio_callback=dummy_audio_callback
            )
            logger.info(f"Unified pipeline created for call: {call_sid}")
        except ValueError as e:
            # Assistant not found or invalid
            logger.warning(f"Invalid assistant ID: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        
        response = CreateCallResponse(
            success=True,
            call_sid=call_sid,
            message="Call initiated successfully with unified pipeline",
            phone_number=request.phone_number,
            assistant_id=request.assistant_id,
            organization_id=request.organization_id
        )
        
        logger.info(f"Call created successfully with unified pipeline. Call SID: {call_sid}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        logger.warning(f"Validation error creating call: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating call: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while creating call"
        )


@router.post('/call-status', response_class=PlainTextResponse)
async def call_status_webhook(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    AccountSid: str = Form(None),
    From: str = Form(None),
    To: str = Form(None),
    CallDuration: str = Form(None),
    Direction: str = Form(None),
):
    """
    Handle Twilio call status webhook.
    
    This endpoint receives status updates from Twilio about call lifecycle events.
    
    Args:
        CallSid: The unique identifier for this call
        CallStatus: The current status of the call
        Other Twilio parameters: Additional information about the call
    
    Returns:
        Empty response with 200 status code
    """
    logger.info(f"Call status webhook received: {CallSid} - {CallStatus}")
    
    try:
        # Handle different call statuses
        if CallStatus == "initiated":
            logger.info(f"Call {CallSid} initiated - unified pipeline should already be created")
            
        elif CallStatus == "ringing":
            logger.info(f"Call {CallSid} is ringing")
            # Nothing to do here - pipeline already prepared
            
        elif CallStatus in ["completed", "busy", "failed", "no-answer", "canceled"]:
            logger.info(f"Call {CallSid} ended with status: {CallStatus}")
            
            # Map Twilio status to our end reason
            end_reason_mapping = {
                "completed": "call_completed",
                "busy": "assistant_hangup", 
                "failed": "system_error",
                "no-answer": "timeout",
                "canceled": "user_cancelled"
            }
            end_reason = end_reason_mapping.get(CallStatus, "system_error")
            
            # Calculate duration if provided
            duration_seconds = None
            if CallDuration:
                try:
                    duration_seconds = int(CallDuration)
                except (ValueError, TypeError):
                    pass
            
            # Update call end status asynchronously
            try:
                from app.services.async_call_service import async_call_service
                await async_call_service.end_call(
                    twilio_call_sid=CallSid,
                    end_reason=end_reason,
                    duration_seconds=duration_seconds
                )
                logger.info(f"Call {CallSid} end status queued for update")
            except Exception as e:
                logger.error(f"Failed to queue call end update for {CallSid}: {e}")
            
            # Shutdown the unified pipeline for this call
            await unified_pipeline_manager.remove_pipeline(CallSid)
            
        else:
            logger.info(f"Call {CallSid} status: {CallStatus}")
        
        # Log additional call information if available
        if CallDuration:
            logger.info(f"Call {CallSid} duration: {CallDuration} seconds")
        
        return ""
        
    except Exception as e:
        logger.error(f"Error processing call status webhook: {e}")
        # Still return 200 to Twilio to acknowledge receipt
        return ""
