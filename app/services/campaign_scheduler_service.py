"""
Campaign Scheduler Service for executing scheduled campaigns.
This service runs every 5 minutes to check for campaigns that need to be executed.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.db.database import get_db
from app.db.models import Campaign, CampaignContact, CampaignStatus, CampaignScheduleType
from app.db.models import PhoneNumber, Assistant
from app.services.twilio_service import twilio_service
from app.services.unified_pipeline import unified_pipeline_manager
from app.db.base_crud import create_call_log

logger = logging.getLogger(__name__)


class CampaignSchedulerService:
    """Service for scheduling and executing campaigns."""
    
    def __init__(self):
        self.is_running = False
        self.scheduler_task = None
    
    async def start_scheduler(self):
        """Start the campaign scheduler that runs every 5 minutes."""
        if self.is_running:
            logger.warning("Campaign scheduler is already running")
            return
        
        self.is_running = True
        logger.info("Starting campaign scheduler service")
        
        # Start the scheduler task
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """Stop the campaign scheduler."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping campaign scheduler service")
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """Main scheduler loop that runs every 5 minutes."""
        while self.is_running:
            try:
                await self._check_and_execute_campaigns()
                # Wait 5 minutes before next check
                await asyncio.sleep(300)  # 300 seconds = 5 minutes
            except asyncio.CancelledError:
                logger.info("Campaign scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in campaign scheduler loop: {e}")
                # Wait 1 minute before retrying on error
                await asyncio.sleep(60)
    
    async def _check_and_execute_campaigns(self):
        """Check for campaigns that need to be executed and execute them."""
        try:
            db = next(get_db())
            try:
                # Find campaigns that are scheduled and ready to execute
                current_time = datetime.utcnow()
                
                # Look for campaigns that are:
                # 1. SCHEDULED type
                # 2. DRAFT status (not yet started)
                # 3. scheduled_at is in the past (time to execute)
                # 4. Have contacts to call
                campaigns_to_execute = db.query(Campaign).filter(
                    and_(
                        Campaign.schedule_type == CampaignScheduleType.SCHEDULED,
                        Campaign.status == CampaignStatus.DRAFT,
                        Campaign.scheduled_at <= current_time,
                        Campaign.contacts.any()  # Has contacts
                    )
                ).all()
                
                logger.info(f"Found {len(campaigns_to_execute)} campaigns ready for execution")
                
                for campaign in campaigns_to_execute:
                    await self._execute_campaign(campaign, db)
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error checking campaigns for execution: {e}")
    
    async def _execute_campaign(self, campaign: Campaign, db: Session):
        """Execute a single campaign by making calls to all contacts."""
        try:
            logger.info(f"Executing campaign: {campaign.name} (ID: {campaign.id})")
            
            # Update campaign status to ACTIVE
            campaign.status = CampaignStatus.ACTIVE
            campaign.started_at = datetime.utcnow()
            db.commit()
            
            # Get all pending contacts for this campaign
            pending_contacts = db.query(CampaignContact).filter(
                and_(
                    CampaignContact.campaign_id == campaign.id,
                    CampaignContact.status == "pending"
                )
            ).all()
            
            logger.info(f"Found {len(pending_contacts)} pending contacts for campaign {campaign.name}")
            
            # Get campaign configuration
            phone_number = db.query(PhoneNumber).filter(
                PhoneNumber.id == campaign.phone_number_id
            ).first()
            
            assistant = db.query(Assistant).filter(
                Assistant.id == campaign.assistant_id
            ).first()
            
            if not phone_number or not assistant:
                logger.error(f"Missing phone number or assistant for campaign {campaign.id}")
                campaign.status = CampaignStatus.CANCELLED
                db.commit()
                return
            
            # Execute calls for each contact
            successful_calls = 0
            failed_calls = 0
            
            for contact in pending_contacts:
                try:
                    # Check if we should make the call based on rate limiting
                    if await self._should_make_call(campaign, db):
                        await self._make_campaign_call(campaign, contact, phone_number, assistant, db)
                        successful_calls += 1
                    else:
                        # Rate limited, schedule for later
                        contact.next_call_attempt = datetime.utcnow() + timedelta(minutes=5)
                        contact.status = "pending"
                        db.commit()
                        
                except Exception as e:
                    logger.error(f"Failed to make call to {contact.phone_number}: {e}")
                    contact.status = "failed"
                    contact.call_attempts += 1
                    contact.last_call_attempt = datetime.utcnow()
                    failed_calls += 1
                    db.commit()
            
            # Update campaign status based on results
            if successful_calls > 0:
                logger.info(f"Campaign {campaign.name} executed: {successful_calls} successful calls, {failed_calls} failed calls")
            else:
                logger.warning(f"Campaign {campaign.name} had no successful calls")
                
        except Exception as e:
            logger.error(f"Error executing campaign {campaign.id}: {e}")
            # Mark campaign as failed
            campaign.status = CampaignStatus.CANCELLED
            db.commit()
    
    async def _should_make_call(self, campaign: Campaign, db: Session) -> bool:
        """Check if we should make a call based on rate limiting."""
        try:
            # Check calls made in the last hour for this campaign
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            recent_calls = db.query(CampaignContact).filter(
                and_(
                    CampaignContact.campaign_id == campaign.id,
                    CampaignContact.last_call_attempt >= one_hour_ago,
                    CampaignContact.status.in_(["called", "completed"])
                )
            ).count()
            
            # Check if we're within the rate limit
            return recent_calls < campaign.max_calls_per_hour
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Default to allowing the call
    
    async def _make_campaign_call(self, campaign: Campaign, contact: CampaignContact, 
                                phone_number: PhoneNumber, assistant: Assistant, db: Session):
        """Make a call for a campaign contact."""
        try:
            logger.info(f"Making call to {contact.phone_number} for campaign {campaign.name}")
            
            # Initialize unified pipeline services if needed
            await unified_pipeline_manager.initialize_services()
            
            # Make the call using Twilio
            call_sid = await twilio_service.make_call(
                contact.phone_number,
                status_callback_url=f"/api/v1/call-status"
            )
            
            # Log the call
            call_data = {
                "organizationId": campaign.organization_id,
                "assistantId": campaign.assistant_id,
                "phoneNumberId": phone_number.id,
                "campaignId": campaign.id,
                "direction": "outbound",
                "status": "initiated",
                "fromNumber": phone_number.phone_number,
                "toNumber": contact.phone_number,
                "startedAt": datetime.utcnow(),
                "twilioCallSid": call_sid
            }
            
            call_log = create_call_log(db, call_data)
            logger.info(f"Campaign call logged with ID: {call_log.id}, Twilio SID: {call_sid}")
            
            # Create unified pipeline for this call
            async def dummy_audio_callback(audio_data: str):
                logger.debug(f"Campaign call audio callback for {call_sid}")
            
            pipeline = await unified_pipeline_manager.create_pipeline(
                call_sid=call_sid,
                assistant_id=campaign.assistant_id,
                audio_callback=dummy_audio_callback
            )
            
            # Update contact status
            contact.status = "called"
            contact.call_attempts += 1
            contact.last_call_attempt = datetime.utcnow()
            contact.next_call_attempt = None
            db.commit()
            
            logger.info(f"Successfully initiated call to {contact.phone_number} for campaign {campaign.name}")
            
        except Exception as e:
            logger.error(f"Failed to make call to {contact.phone_number}: {e}")
            # Update contact status to failed
            contact.status = "failed"
            contact.call_attempts += 1
            contact.last_call_attempt = datetime.utcnow()
            contact.next_call_attempt = datetime.utcnow() + timedelta(minutes=campaign.retry_delay_minutes)
            db.commit()
            raise
    
    async def execute_campaign_now(self, campaign_id: str) -> bool:
        """Manually execute a campaign immediately (for NOW type campaigns)."""
        try:
            db = next(get_db())
            try:
                campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
                if not campaign:
                    logger.error(f"Campaign {campaign_id} not found")
                    return False
                
                if campaign.schedule_type != CampaignScheduleType.NOW:
                    logger.error(f"Campaign {campaign_id} is not a NOW type campaign")
                    return False
                
                if campaign.status != CampaignStatus.DRAFT:
                    logger.error(f"Campaign {campaign_id} is not in DRAFT status")
                    return False
                
                await self._execute_campaign(campaign, db)
                return True
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error executing campaign {campaign_id} now: {e}")
            return False
    
    async def get_campaign_status(self, campaign_id: str) -> Optional[dict]:
        """Get the current status of a campaign."""
        try:
            db = next(get_db())
            try:
                campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
                if not campaign:
                    return None
                
                # Get contact statistics
                total_contacts = db.query(CampaignContact).filter(
                    CampaignContact.campaign_id == campaign.id
                ).count()
                
                pending_contacts = db.query(CampaignContact).filter(
                    and_(
                        CampaignContact.campaign_id == campaign.id,
                        CampaignContact.status == "pending"
                    )
                ).count()
                
                called_contacts = db.query(CampaignContact).filter(
                    and_(
                        CampaignContact.campaign_id == campaign.id,
                        CampaignContact.status == "called"
                    )
                ).count()
                
                completed_contacts = db.query(CampaignContact).filter(
                    and_(
                        CampaignContact.campaign_id == campaign.id,
                        CampaignContact.status == "completed"
                    )
                ).count()
                
                failed_contacts = db.query(CampaignContact).filter(
                    and_(
                        CampaignContact.campaign_id == campaign.id,
                        CampaignContact.status == "failed"
                    )
                ).count()
                
                return {
                    "campaign_id": str(campaign.id),
                    "name": campaign.name,
                    "status": campaign.status.value,
                    "schedule_type": campaign.schedule_type.value,
                    "scheduled_at": campaign.scheduled_at.isoformat() if campaign.scheduled_at else None,
                    "started_at": campaign.started_at.isoformat() if campaign.started_at else None,
                    "completed_at": campaign.completed_at.isoformat() if campaign.completed_at else None,
                    "total_contacts": total_contacts,
                    "pending_contacts": pending_contacts,
                    "called_contacts": called_contacts,
                    "completed_contacts": completed_contacts,
                    "failed_contacts": failed_contacts,
                    "success_rate": (completed_contacts / total_contacts * 100) if total_contacts > 0 else 0
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting campaign status for {campaign_id}: {e}")
            return None


# Global campaign scheduler instance
campaign_scheduler = CampaignSchedulerService()
