"""
Campaign API routes.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud.campaign_crud import (
    create_campaign, get_campaign, get_campaigns, get_campaigns_count, update_campaign, delete_campaign,
    start_campaign, pause_campaign, resume_campaign, stop_campaign,
    create_campaign_contact, get_campaign_contacts, update_campaign_contact
)
from app.models.campaign_schemas import (
    Campaign, CampaignCreate, CampaignUpdate, CampaignListResponse,
    CampaignContact, CampaignContactCreate, CampaignContactUpdate,
    CampaignActionRequest, CampaignActionResponse, CSVUploadRequest, CSVUploadResponse,
    CampaignAnalytics
)
from app.core.auth import require_read_permission, require_write_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/organizations/{organization_id}/campaigns", tags=["campaigns"])


@router.get("/", response_model=CampaignListResponse)
async def list_campaigns(
    organization_id: uuid.UUID,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search campaigns"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get campaigns for an organization with pagination and filtering.
    """
    try:
        organization, membership = org_data
        
        # Convert status string to enum if provided
        status_enum = None
        if status:
            try:
                from app.db.models import CampaignStatus
                status_enum = CampaignStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        skip = (page - 1) * per_page
        campaigns = get_campaigns(
            db=db,
            organization_id=organization.id,
            skip=skip,
            limit=per_page,
            status=status_enum,
            search=search
        )
        
        total = get_campaigns_count(
            db=db,
            organization_id=organization.id,
            status=status_enum,
            search=search
        )
        
        
        return CampaignListResponse(
            items=campaigns,
            total=total,
            page=page,
            page_size=per_page,
            total_pages=(total + per_page - 1) // per_page
        )
        
    except Exception as e:
        logger.error(f"Error fetching campaigns: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch campaigns")


@router.get("/{campaign_id}", response_model=Campaign)
async def get_campaign_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get a specific campaign by ID.
    """
    try:
        organization, membership = org_data
        
        campaign = get_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        
        return campaign
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching campaign: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch campaign")


@router.post("/", response_model=Campaign)
async def create_campaign_endpoint(
    campaign_data: CampaignCreate,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Create a new campaign with contacts from CSV data.
    """
    try:
        organization, membership = org_data
        
        campaign = create_campaign(
            db=db,
            campaign_data=campaign_data,
            organization_id=organization.id
        )
        
        logger.info(f"Created campaign {campaign.id} for organization {organization.id}")
        return campaign
        
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create campaign")


@router.put("/{campaign_id}", response_model=Campaign)
async def update_campaign_endpoint(
    campaign_id: uuid.UUID,
    campaign_data: CampaignUpdate,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Update an existing campaign.
    """
    try:
        organization, membership = org_data
        
        campaign = update_campaign(
            db=db,
            campaign_id=campaign_id,
            organization_id=organization.id,
            campaign_data=campaign_data
        )
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        logger.info(f"Updated campaign {campaign_id} for organization {organization.id}")
        return campaign
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update campaign")


@router.delete("/{campaign_id}")
async def delete_campaign_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Delete a campaign and all its contacts.
    """
    try:
        organization, membership = org_data
        
        success = delete_campaign(db, campaign_id, organization.id)
        if not success:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        logger.info(f"Deleted campaign {campaign_id} for organization {organization.id}")
        return {"message": "Campaign deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete campaign")


@router.post("/{campaign_id}/start", response_model=CampaignActionResponse)
async def start_campaign_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Start a campaign.
    """
    try:
        organization, membership = org_data
        
        campaign = start_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=400, detail="Campaign cannot be started")
        
        logger.info(f"Started campaign {campaign_id} for organization {organization.id}")
        return CampaignActionResponse(
            success=True,
            message="Campaign started successfully",
            campaign=campaign
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to start campaign")


@router.post("/{campaign_id}/pause", response_model=CampaignActionResponse)
async def pause_campaign_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Pause a campaign.
    """
    try:
        organization, membership = org_data
        
        campaign = pause_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=400, detail="Campaign cannot be paused")
        
        logger.info(f"Paused campaign {campaign_id} for organization {organization.id}")
        return CampaignActionResponse(
            success=True,
            message="Campaign paused successfully",
            campaign=campaign
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to pause campaign")


@router.post("/{campaign_id}/resume", response_model=CampaignActionResponse)
async def resume_campaign_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Resume a paused campaign.
    """
    try:
        organization, membership = org_data
        
        campaign = resume_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=400, detail="Campaign cannot be resumed")
        
        logger.info(f"Resumed campaign {campaign_id} for organization {organization.id}")
        return CampaignActionResponse(
            success=True,
            message="Campaign resumed successfully",
            campaign=campaign
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to resume campaign")


@router.post("/{campaign_id}/stop", response_model=CampaignActionResponse)
async def stop_campaign_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Stop a campaign.
    """
    try:
        organization, membership = org_data
        
        campaign = stop_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=400, detail="Campaign cannot be stopped")
        
        logger.info(f"Stopped campaign {campaign_id} for organization {organization.id}")
        return CampaignActionResponse(
            success=True,
            message="Campaign stopped successfully",
            campaign=campaign
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to stop campaign")


@router.get("/{campaign_id}/contacts", response_model=List[CampaignContact])
async def get_campaign_contacts_endpoint(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=1000, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by contact status"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get contacts for a campaign.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = get_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Convert status string to enum if provided
        status_enum = None
        if status:
            try:
                from app.db.models import CampaignContactStatus
                status_enum = CampaignContactStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        skip = (page - 1) * per_page
        contacts = get_campaign_contacts(
            db=db,
            campaign_id=campaign_id,
            skip=skip,
            limit=per_page,
            status=status_enum
        )
        
        return contacts
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching campaign contacts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch campaign contacts")




@router.post("/{campaign_id}/upload-csv", response_model=CSVUploadResponse)
async def upload_csv_to_campaign(
    campaign_id: uuid.UUID,
    organization_id: uuid.UUID,
    csv_data: CSVUploadRequest,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """
    Upload CSV data to an existing campaign.
    """
    try:
        organization, membership = org_data
        
        # Verify campaign exists and belongs to organization
        campaign = get_campaign(db, campaign_id, organization.id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        contacts_created = 0
        contacts_updated = 0
        errors = []
        
        # Process CSV data
        for row_data in csv_data.csv_data:
            try:
                contact_data = CampaignContactCreate(
                    phone_number=row_data.get(csv_data.headers.get('phone_number', 'phone_number'), ''),
                    name=row_data.get(csv_data.headers.get('name', 'name'), ''),
                    email=row_data.get(csv_data.headers.get('email', 'email'), '')
                )
                
                # Check if contact already exists
                existing_contact = db.query(CampaignContact).filter(
                    CampaignContact.campaign_id == campaign_id,
                    CampaignContact.phone_number == contact_data.phone_number
                ).first()
                
                if existing_contact:
                    # Update existing contact
                    update_data = CampaignContactUpdate(
                        name=contact_data.name,
                        email=contact_data.email
                    )
                    update_campaign_contact(db, existing_contact.id, update_data)
                    contacts_updated += 1
                else:
                    # Create new contact
                    create_campaign_contact(db, contact_data, campaign_id)
                    contacts_created += 1
                    
            except Exception as e:
                errors.append(f"Error processing contact {row_data}: {str(e)}")
        
        logger.info(f"Uploaded CSV to campaign {campaign_id}: {contacts_created} created, {contacts_updated} updated")
        
        return CSVUploadResponse(
            success=True,
            message=f"CSV uploaded successfully: {contacts_created} contacts created, {contacts_updated} contacts updated",
            contacts_created=contacts_created,
            contacts_updated=contacts_updated,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading CSV to campaign: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to upload CSV to campaign")
