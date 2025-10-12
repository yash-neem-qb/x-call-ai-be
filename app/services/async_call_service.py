"""
Async Call Service for handling call updates without blocking the main pipeline.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.base_crud import (
    update_call_by_twilio_sid, 
    add_transcript_message_by_twilio_sid,
    get_call_by_twilio_sid
)
from app.db.models import CallStatus, CallEndReason

logger = logging.getLogger(__name__)


class AsyncCallService:
    """Service for handling async call updates to avoid pipeline latency."""
    
    def __init__(self):
        self.update_queue = asyncio.Queue()
        self.transcript_queue = asyncio.Queue()
        self.is_running = False
        self.worker_task = None
    
    async def start(self):
        """Start the async worker."""
        if not self.is_running:
            self.is_running = True
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("Async call service started")
    
    async def stop(self):
        """Stop the async worker."""
        if self.is_running:
            self.is_running = False
            if self.worker_task:
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass
            logger.info("Async call service stopped")
    
    async def _worker(self):
        """Background worker to process call updates."""
        while self.is_running:
            try:
                # Process call updates
                try:
                    update_task = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                    await self._process_call_update(update_task)
                except asyncio.TimeoutError:
                    pass
                
                # Process transcript updates
                try:
                    transcript_task = await asyncio.wait_for(self.transcript_queue.get(), timeout=1.0)
                    await self._process_transcript_update(transcript_task)
                except asyncio.TimeoutError:
                    pass
                
            except Exception as e:
                logger.error(f"Error in async call service worker: {e}")
                await asyncio.sleep(1)
    
    async def _process_call_update(self, update_data: Dict[str, Any]):
        """Process a call update in a separate thread."""
        try:
            # Run database operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._update_call_sync, update_data)
        except Exception as e:
            logger.error(f"Error processing call update: {e}")
    
    async def _process_transcript_update(self, transcript_data: Dict[str, Any]):
        """Process a transcript update in a separate thread."""
        try:
            # Run database operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._add_transcript_sync, transcript_data)
        except Exception as e:
            logger.error(f"Error processing transcript update: {e}")
    
    def _update_call_sync(self, update_data: Dict[str, Any]):
        """Synchronous call update (runs in thread pool)."""
        try:
            db = next(get_db())
            try:
                twilio_call_sid = update_data.get("twilio_call_sid")
                if twilio_call_sid:
                    # Remove twilio_call_sid from update data as it's used for lookup
                    update_data_copy = {k: v for k, v in update_data.items() if k != "twilio_call_sid"}
                    
                    # For web calls, the twilio_call_sid is actually the session_id
                    # Check if it's a web call by the prefix
                    if twilio_call_sid.startswith("webrtc_") or twilio_call_sid.startswith("chat_"):
                        # For web calls, we need to find by session_id instead
                        from app.db.base_crud import get_call_by_session_id
                        call = get_call_by_session_id(db, twilio_call_sid)
                        if call:
                            # Update the call directly
                            for key, value in update_data_copy.items():
                                if hasattr(call, key) and value is not None:
                                    setattr(call, key, value)
                            db.commit()
                            db.refresh(call)
                            logger.info(f"Updated web call by session ID: {twilio_call_sid}")
                        else:
                            logger.warning(f"Web call not found with session ID: {twilio_call_sid}")
                    else:
                        # Regular phone call - use twilio_call_sid
                        update_call_by_twilio_sid(db, twilio_call_sid, update_data_copy)
                else:
                    logger.warning("No twilio_call_sid provided for call update")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error in sync call update: {e}")
    
    def _add_transcript_sync(self, transcript_data: Dict[str, Any]):
        """Synchronous transcript update (runs in thread pool)."""
        try:
            db = next(get_db())
            try:
                twilio_call_sid = transcript_data.get("twilio_call_sid")
                message_data = transcript_data.get("message_data")
                
                logger.info(f"🔍 Processing transcript for call_sid: {twilio_call_sid}, message: {message_data}")
                
                if twilio_call_sid and message_data:
                    # For web calls, the twilio_call_sid is actually the session_id
                    if twilio_call_sid.startswith("webrtc_") or twilio_call_sid.startswith("chat_"):
                        # For web calls, find by session_id and add transcript directly
                        from app.db.base_crud import get_call_by_session_id
                        call = get_call_by_session_id(db, twilio_call_sid)
                        logger.info(f"🔍 Lookup by session_id '{twilio_call_sid}' found call: {call.id if call else 'None'}")
                        
                        if call:
                            # Initialize transcript_data if it doesn't exist
                            if call.transcript_data is None:
                                call.transcript_data = []
                                logger.info(f"🔍 Initialized empty transcript_data for call {call.id}")
                            
                            # Add the new message
                            call.transcript_data.append(message_data)
                            logger.info(f"🔍 Before commit - transcript_data length: {len(call.transcript_data)}, content: {call.transcript_data}")
                            
                            # Mark the field as modified so SQLAlchemy knows to update it
                            from sqlalchemy.orm.attributes import flag_modified
                            flag_modified(call, 'transcript_data')
                            
                            db.commit()
                            db.refresh(call)
                            logger.info(f"🔍 After commit and refresh - transcript_data length: {len(call.transcript_data)}, content: {call.transcript_data}")
                            logger.info(f"✅ Added transcript message to web call by session ID: {twilio_call_sid}, total messages: {len(call.transcript_data)}")
                        else:
                            logger.warning(f"❌ Web call not found with session ID: {twilio_call_sid}")
                            # Try to find by twilio_call_sid as fallback
                            from app.db.base_crud import get_call_by_twilio_sid
                            fallback_call = get_call_by_twilio_sid(db, twilio_call_sid)
                            logger.info(f"🔍 Fallback lookup by twilio_call_sid '{twilio_call_sid}' found call: {fallback_call.id if fallback_call else 'None'}")
                    else:
                        # Regular phone call - use existing function
                        add_transcript_message_by_twilio_sid(db, twilio_call_sid, message_data)
                else:
                    logger.warning("❌ Missing twilio_call_sid or message_data for transcript update")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"❌ Error in sync transcript update: {e}")
    
    async def update_call_status(self, twilio_call_sid: str, status: str, **kwargs):
        """Update call status asynchronously."""
        # Convert status string to enum if it's a valid status
        try:
            status_enum = CallStatus(status)
        except ValueError:
            # If it's not a valid status, use as string (for backward compatibility)
            status_enum = status
        
        update_data = {
            "twilio_call_sid": twilio_call_sid,
            "status": status_enum,
            "updated_at": datetime.utcnow()
        }
        
        # Add any additional fields
        update_data.update(kwargs)
        
        await self.update_queue.put(update_data)
        logger.debug(f"Queued call status update for {twilio_call_sid}: {status}")
    
    async def update_call_metadata(self, twilio_call_sid: str, **metadata):
        """Update call metadata asynchronously."""
        update_data = {
            "twilio_call_sid": twilio_call_sid,
            "updated_at": datetime.utcnow()
        }
        update_data.update(metadata)
        
        await self.update_queue.put(update_data)
        logger.debug(f"Queued call metadata update for {twilio_call_sid}")
    
    async def add_transcript_message(self, twilio_call_sid: str, speaker: str, message: str, 
                                   confidence: float = 1.0, **kwargs):
        """Add a transcript message asynchronously."""
        # Use current UTC time for timestamp
        current_time = datetime.utcnow()
        message_data = {
            "speaker": speaker,
            "message": message,
            "timestamp": current_time.isoformat() + "Z",
            "confidence": confidence
        }
        message_data.update(kwargs)
        
        transcript_data = {
            "twilio_call_sid": twilio_call_sid,
            "message_data": message_data
        }
        
        await self.transcript_queue.put(transcript_data)
        logger.info(f"🔍 Queued transcript message for {twilio_call_sid}: {speaker} - '{message[:30]}...'")
    
    async def end_call(self, twilio_call_sid: str, end_reason: str, duration_seconds: Optional[int] = None):
        """End a call and update all relevant fields."""
        # Convert string values to enum values
        status_enum = CallStatus.COMPLETED
        end_reason_enum = CallEndReason(end_reason) if end_reason else None
        
        update_data = {
            "twilio_call_sid": twilio_call_sid,
            "status": status_enum,
            "end_reason": end_reason_enum,
            "ended_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if duration_seconds is not None:
            update_data["duration_seconds"] = duration_seconds
        
        await self.update_queue.put(update_data)
        logger.info(f"Queued call end for {twilio_call_sid}: {end_reason}")
        
        # Trigger call analysis asynchronously
        asyncio.create_task(self._analyze_call_async(twilio_call_sid))
    
    async def _analyze_call_async(self, twilio_call_sid: str):
        """Analyze call asynchronously after it ends."""
        try:
            logger.info(f"Starting call analysis for {twilio_call_sid}")
            
            # Wait a bit for the call end update to be processed
            await asyncio.sleep(2)
            
            # Get call data for analysis
            call_data = await self._get_call_for_analysis(twilio_call_sid)
            if not call_data:
                logger.warning(f"Could not retrieve call data for analysis: {twilio_call_sid}")
                return
            
            # Import here to avoid circular imports
            from app.services.call_analysis_service import call_analysis_service
            
            # Perform call analysis
            analysis_result = await call_analysis_service.analyze_call(
                transcript_data=call_data["transcript_data"],
                system_prompt=call_data["system_prompt"],
                assistant_id=call_data.get("assistant_id")
            )
            
            # Update call with analysis results
            analysis_update = {
                "twilio_call_sid": twilio_call_sid,
                "call_success": analysis_result.call_success,
                "call_summary": analysis_result.call_summary,
                "sentiment_score": analysis_result.sentiment_score,
                "detailed_analysis": analysis_result.detailed_analysis,
                "analysis_completed": True,
                "updated_at": datetime.utcnow()
            }
            
            await self.update_queue.put(analysis_update)
            logger.info(f"Call analysis completed for {twilio_call_sid} - Success: {analysis_result.call_success}, Sentiment: {analysis_result.sentiment_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error in call analysis for {twilio_call_sid}: {e}")
            # Mark analysis as failed
            try:
                failed_update = {
                    "twilio_call_sid": twilio_call_sid,
                    "analysis_completed": True,
                    "call_summary": "Analysis failed due to error",
                    "updated_at": datetime.utcnow()
                }
                await self.update_queue.put(failed_update)
            except Exception as update_error:
                logger.error(f"Failed to update call with analysis failure: {update_error}")
    
    async def _get_call_for_analysis(self, twilio_call_sid: str) -> Optional[Dict[str, Any]]:
        """Get call data needed for analysis."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_call_for_analysis_sync, twilio_call_sid)
        except Exception as e:
            logger.error(f"Error getting call data for analysis: {e}")
            return None
    
    def _get_call_for_analysis_sync(self, twilio_call_sid: str) -> Optional[Dict[str, Any]]:
        """Synchronously get call data for analysis."""
        try:
            db = next(get_db())
            try:
                # Get call by twilio_call_sid or session_id
                if twilio_call_sid.startswith("webrtc_") or twilio_call_sid.startswith("chat_"):
                    from app.db.base_crud import get_call_by_session_id
                    call = get_call_by_session_id(db, twilio_call_sid)
                else:
                    call = get_call_by_twilio_sid(db, twilio_call_sid)
                
                if not call:
                    logger.warning(f"Call not found for analysis: {twilio_call_sid}")
                    return None
                
                # Get system prompt from assistant
                system_prompt = "You are a helpful voice assistant. Be concise and conversational."
                if call.assistant_id:
                    from app.db.crud import get_assistant
                    assistant = get_assistant(db, call.assistant_id)
                    if assistant:
                        # Check if assistant has model configuration
                        model_config = getattr(assistant, 'model', None) or getattr(assistant, 'model_config', None)
                        if model_config and isinstance(model_config, dict) and model_config.get("messages"):
                            for message in model_config["messages"]:
                                if message.get("role") == "system":
                                    system_prompt = message.get("content", system_prompt)
                                    break
                
                return {
                    "transcript_data": call.transcript_data or [],
                    "system_prompt": system_prompt,
                    "assistant_id": str(call.assistant_id) if call.assistant_id else None
                }
                
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error in sync call data retrieval: {e}")
            return None


# Global instance
async_call_service = AsyncCallService()
