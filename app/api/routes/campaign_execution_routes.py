"""
API routes for campaign execution and management.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Campaign, CampaignContact, CampaignStatus, CampaignScheduleType
from app.db.models import PhoneNumber, Assistant
from app.models.campaign_schemas import Campaign, CampaignContact
from app.services.campaign_scheduler_service import campaign_scheduler
from app.core.auth import require_read_permission, require_write_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/campaigns", tags=["campaign-execution"])


@router.post("/{campaign_id}/execute", response_model=dict)
async def execute_campaign_now(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Manually execute a campaign immediately.
    
    This endpoint allows you to start a campaign that is in DRAFT status
    and has schedule_type = NOW.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization.id
        ).first()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        if campaign.schedule_type != CampaignScheduleType.NOW:
            raise HTTPException(
                status_code=400, 
                detail="Only NOW type campaigns can be executed immediately"
            )
        
        if campaign.status != CampaignStatus.DRAFT:
            raise HTTPException(
                status_code=400, 
                detail="Only DRAFT campaigns can be executed"
            )
        
        # Check if campaign has contacts
        contact_count = db.query(CampaignContact).filter(
            CampaignContact.campaign_id == campaign_id
        ).count()
        
        if contact_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="Campaign has no contacts to call"
            )
        
        # Execute the campaign
        success = await campaign_scheduler.execute_campaign_now(str(campaign_id))
        
        if success:
            logger.info(f"Campaign {campaign_id} executed successfully")
            return {
                "success": True,
                "message": "Campaign execution started successfully",
                "campaign_id": str(campaign_id),
                "contact_count": contact_count
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to execute campaign"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing campaign {campaign_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while executing campaign"
        )


@router.get("/{campaign_id}/status", response_model=dict)
async def get_campaign_execution_status(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get the current execution status of a campaign.
    
    Returns detailed information about the campaign's progress,
    including contact statistics and call results.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization.id
        ).first()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Get campaign status from scheduler service
        status = await campaign_scheduler.get_campaign_status(str(campaign_id))
        
        if not status:
            raise HTTPException(
                status_code=500, 
                detail="Failed to get campaign status"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign status for {campaign_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while getting campaign status"
        )


@router.post("/{campaign_id}/pause", response_model=dict)
async def pause_campaign(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Pause an active campaign.
    
    This will stop the campaign from making new calls but won't
    affect calls that are already in progress.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization.id
        ).first()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        if campaign.status != CampaignStatus.ACTIVE:
            raise HTTPException(
                status_code=400, 
                detail="Only ACTIVE campaigns can be paused"
            )
        
        # Update campaign status
        campaign.status = CampaignStatus.PAUSED
        campaign.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Campaign {campaign_id} paused successfully")
        
        return {
            "success": True,
            "message": "Campaign paused successfully",
            "campaign_id": str(campaign_id),
            "status": "PAUSED"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing campaign {campaign_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while pausing campaign"
        )


@router.post("/{campaign_id}/resume", response_model=dict)
async def resume_campaign(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Resume a paused campaign.
    
    This will restart the campaign and continue making calls
    to remaining contacts.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization.id
        ).first()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        if campaign.status != CampaignStatus.PAUSED:
            raise HTTPException(
                status_code=400, 
                detail="Only PAUSED campaigns can be resumed"
            )
        
        # Update campaign status
        campaign.status = CampaignStatus.ACTIVE
        campaign.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Campaign {campaign_id} resumed successfully")
        
        return {
            "success": True,
            "message": "Campaign resumed successfully",
            "campaign_id": str(campaign_id),
            "status": "ACTIVE"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming campaign {campaign_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while resuming campaign"
        )


@router.post("/{campaign_id}/stop", response_model=dict)
async def stop_campaign(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Stop a campaign completely.
    
    This will mark the campaign as COMPLETED and stop all
    future call attempts.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization.id
        ).first()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        if campaign.status not in [CampaignStatus.ACTIVE, CampaignStatus.PAUSED]:
            raise HTTPException(
                status_code=400, 
                detail="Only ACTIVE or PAUSED campaigns can be stopped"
            )
        
        # Update campaign status
        campaign.status = CampaignStatus.COMPLETED
        campaign.completed_at = datetime.utcnow()
        campaign.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Campaign {campaign_id} stopped successfully")
        
        return {
            "success": True,
            "message": "Campaign stopped successfully",
            "campaign_id": str(campaign_id),
            "status": "COMPLETED"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping campaign {campaign_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while stopping campaign"
        )


@router.get("/{campaign_id}/contacts", response_model=List[dict])
async def get_campaign_contacts_with_status(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    status: Optional[str] = Query(None, description="Filter by contact status"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=1000, description="Items per page"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get contacts for a campaign with their current status.
    
    This endpoint provides detailed information about each contact
    in the campaign, including call attempts and status.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization.id
        ).first()
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Build query
        query = db.query(CampaignContact).filter(
            CampaignContact.campaign_id == campaign_id
        )
        
        # Apply status filter if provided
        if status:
            query = query.filter(CampaignContact.status == status)
        
        # Apply pagination
        offset = (page - 1) * per_page
        contacts = query.offset(offset).limit(per_page).all()
        
        # Convert to response format
        contact_list = []
        for contact in contacts:
            contact_list.append({
                "id": str(contact.id),
                "phone_number": contact.phone_number,
                "name": contact.name,
                "email": contact.email,
                "status": contact.status,
                "call_attempts": contact.call_attempts,
                "last_call_attempt": contact.last_call_attempt.isoformat() if contact.last_call_attempt else None,
                "next_call_attempt": contact.next_call_attempt.isoformat() if contact.next_call_attempt else None,
                "created_at": contact.created_at.isoformat(),
                "updated_at": contact.updated_at.isoformat()
            })
        
        return contact_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign contacts for {campaign_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while getting campaign contacts"
        )


@router.get("/scheduler/status", response_model=dict)
async def get_scheduler_status(
    org_data: tuple = Depends(require_read_permission())
):
    """
    Get the current status of the campaign scheduler service.
    
    This endpoint provides information about whether the scheduler
    is running and its current state.
    """
    try:
        return {
            "scheduler_running": campaign_scheduler.is_running,
            "message": "Campaign scheduler is running" if campaign_scheduler.is_running else "Campaign scheduler is not running"
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while getting scheduler status"
        )
