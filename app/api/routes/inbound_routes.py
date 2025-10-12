"""
API routes for inbound call configuration.
Handles webhook setup for incoming calls.
"""

import logging
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.twilio_service import twilio_service
from app.config.settings import settings
from app.db.database import get_db
from app.db.models import PhoneNumber
from app.db.crud import create_call_log, update_call_log
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/inbound", tags=["inbound"])


class InboundWebhookConfigRequest(BaseModel):
    """Request model for configuring inbound webhooks."""
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    PHONE_NUMBER_FROM: str


class InboundWebhookConfigResponse(BaseModel):
    """Response model for webhook configuration."""
    success: bool
    message: str
    phone_number: str
    voice_url: str
    status_callback_url: str
    friendly_name: str
    sid: str




@router.post("/webhook")
async def handle_inbound_call(request: Request):
    """
    Handle incoming call webhook from Twilio.
    
    This endpoint receives the webhook when someone calls the configured phone number.
    """
    try:
        # Get form data from Twilio
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")
        
        logger.info(f"Incoming call from {from_number} to {to_number}, Call SID: {call_sid}")
        
        # Log the inbound call
        try:
            # Find the phone number and its associated assistant
            db = next(get_db())
            try:
                phone_number = db.query(PhoneNumber).filter(PhoneNumber.phone_number == to_number).first()
                assistant_id = phone_number.assistant_id if phone_number else None
                organization_id = phone_number.organization_id if phone_number else None
                
                if assistant_id and organization_id:
                    call_data = {
                        "organizationId": organization_id,
                        "assistantId": assistant_id,
                        "phoneNumberId": phone_number.id,
                        "direction": "inbound",
                        "status": "initiated",
                        "fromNumber": from_number,
                        "toNumber": to_number,
                        "startedAt": datetime.utcnow()
                    }
                    call_log = create_call_log(db, call_data)
                    logger.info(f"Inbound call logged with ID: {call_log.id}")
                else:
                    logger.warning(f"No assistant found for phone number {to_number}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to log inbound call: {e}")
        
        # Create TwiML response to connect the call to our media stream
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{settings.domain}/media-stream" />
    </Connect>
    <Say>Thank you for calling. Goodbye.</Say>
</Response>"""
        
        return JSONResponse(
            content={"twiml": twiml_response},
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Error handling inbound call webhook: {e}")
        # Return a simple TwiML response in case of error
        error_twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, we're experiencing technical difficulties. Please try again later.</Say>
</Response>"""
        return JSONResponse(
            content={"twiml": error_twiml},
            media_type="application/xml"
        )


@router.post("/status")
async def handle_call_status(request: Request):
    """
    Handle call status webhook from Twilio.
    
    This endpoint receives status updates about the call (ringing, answered, completed, etc.).
    """
    try:
        # Get form data from Twilio
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        from_number = form_data.get("From")
        to_number = form_data.get("To")
        
        logger.info(f"Call status update: {call_sid} - {call_status} (from: {from_number}, to: {to_number})")
        
        # Handle different call statuses
        if call_status == "completed":
            logger.info(f"Call {call_sid} completed")
            
            # Log the call end (we need to find the call log by CallSid)
            try:
                # For now, we'll need to implement a way to find the call log
                # This could be done by storing the mapping or using a different approach
                logger.info(f"Inbound call {call_sid} completed - call logging would be updated here")
                # TODO: Implement call log update by CallSid
            except Exception as e:
                logger.error(f"Failed to log inbound call end for {call_sid}: {e}")
            
            # Here you could add cleanup logic if needed
        
        return JSONResponse(content={"status": "received"})
        
    except Exception as e:
        logger.error(f"Error handling call status webhook: {e}")
        return JSONResponse(content={"status": "error"}, status_code=500)
