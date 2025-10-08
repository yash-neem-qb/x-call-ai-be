"""
CRUD operations for campaigns and campaign contacts.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError

from app.db.models import Campaign, CampaignContact, CampaignStatus, CampaignScheduleType, CampaignContactStatus
from app.models.campaign_schemas import (
    CampaignCreate, CampaignUpdate, CampaignContactCreate, CampaignContactUpdate,
    CampaignStats
)


def create_campaign(db: Session, campaign_data: CampaignCreate, organization_id: uuid.UUID) -> Campaign:
    """Create a new campaign with contacts."""
    try:
        # Create campaign
        campaign = Campaign(
            organization_id=organization_id,
            name=campaign_data.name,
            description=campaign_data.description,
            assistant_id=campaign_data.assistant_id,
            phone_number_id=campaign_data.phone_number_id,
            schedule_type=campaign_data.schedule_type,
            scheduled_at=campaign_data.scheduled_at,
            max_calls_per_hour=campaign_data.max_calls_per_hour,
            retry_failed_calls=campaign_data.retry_failed_calls,
            max_retries=campaign_data.max_retries,
            retry_delay_minutes=campaign_data.retry_delay_minutes,
            status=CampaignStatus.DRAFT
        )
        
        db.add(campaign)
        db.flush()  # Get the campaign ID
        
        # Create contacts if provided
        if campaign_data.contacts:
            for contact_data in campaign_data.contacts:
                contact = CampaignContact(
                    campaign_id=campaign.id,
                    phone_number=contact_data.phone_number,
                    name=contact_data.name,
                    email=contact_data.email,
                    custom_field_1=contact_data.custom_field_1,
                    custom_field_2=contact_data.custom_field_2,
                    custom_field_3=contact_data.custom_field_3,
                    custom_field_4=contact_data.custom_field_4,
                    custom_field_5=contact_data.custom_field_5,
                    status=CampaignContactStatus.PENDING
                )
                db.add(contact)
        
        # Process CSV data if provided
        if campaign_data.csv_data and campaign_data.csv_headers:
            for row_data in campaign_data.csv_data:
                contact = CampaignContact(
                    campaign_id=campaign.id,
                    phone_number=row_data.get(campaign_data.csv_headers.get('phone_number', 'phone_number'), ''),
                    name=row_data.get(campaign_data.csv_headers.get('name', 'name'), ''),
                    email=row_data.get(campaign_data.csv_headers.get('email', 'email'), ''),
                    custom_field_1=row_data.get(campaign_data.csv_headers.get('custom_field_1', ''), ''),
                    custom_field_2=row_data.get(campaign_data.csv_headers.get('custom_field_2', ''), ''),
                    custom_field_3=row_data.get(campaign_data.csv_headers.get('custom_field_3', ''), ''),
                    custom_field_4=row_data.get(campaign_data.csv_headers.get('custom_field_4', ''), ''),
                    custom_field_5=row_data.get(campaign_data.csv_headers.get('custom_field_5', ''), ''),
                    status=CampaignContactStatus.PENDING
                )
                db.add(contact)
        
        db.commit()
        db.refresh(campaign)
        return campaign
        
    except IntegrityError as e:
        db.rollback()
        raise e


def get_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Campaign]:
    """Get a campaign by ID."""
    return db.query(Campaign).filter(
        and_(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization_id
        )
    ).first()


def get_campaigns(
    db: Session, 
    organization_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    status: Optional[CampaignStatus] = None,
    search: Optional[str] = None
) -> List[Campaign]:
    """Get campaigns for an organization with pagination and filtering."""
    query = db.query(Campaign).filter(Campaign.organization_id == organization_id)
    
    # Apply filters
    if status:
        query = query.filter(Campaign.status == status)
    
    if search:
        query = query.filter(
            or_(
                Campaign.name.ilike(f"%{search}%"),
                Campaign.description.ilike(f"%{search}%")
            )
        )
    
    return query.order_by(desc(Campaign.created_at)).offset(skip).limit(limit).all()


def get_campaigns_count(
    db: Session, 
    organization_id: uuid.UUID,
    status: Optional[CampaignStatus] = None,
    search: Optional[str] = None
) -> int:
    """Get total count of campaigns for an organization."""
    query = db.query(Campaign).filter(Campaign.organization_id == organization_id)
    
    if status:
        query = query.filter(Campaign.status == status)
    
    if search:
        query = query.filter(
            or_(
                Campaign.name.ilike(f"%{search}%"),
                Campaign.description.ilike(f"%{search}%")
            )
        )
    
    return query.count()


def update_campaign(
    db: Session, 
    campaign_id: uuid.UUID, 
    organization_id: uuid.UUID,
    campaign_data: CampaignUpdate
) -> Optional[Campaign]:
    """Update a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign:
        return None
    
    update_data = campaign_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(campaign, field, value)
    
    campaign.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(campaign)
    return campaign


def delete_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
    """Delete a campaign and all its contacts."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign:
        return False
    
    db.delete(campaign)
    db.commit()
    return True


def start_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Campaign]:
    """Start a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status != CampaignStatus.DRAFT:
        return None
    
    campaign.status = CampaignStatus.ACTIVE
    campaign.started_at = datetime.utcnow()
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def pause_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Campaign]:
    """Pause a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status != CampaignStatus.ACTIVE:
        return None
    
    campaign.status = CampaignStatus.PAUSED
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def resume_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Campaign]:
    """Resume a paused campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status != CampaignStatus.PAUSED:
        return None
    
    campaign.status = CampaignStatus.ACTIVE
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def stop_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Campaign]:
    """Stop a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status not in [CampaignStatus.ACTIVE, CampaignStatus.PAUSED]:
        return None
    
    campaign.status = CampaignStatus.COMPLETED
    campaign.completed_at = datetime.utcnow()
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def get_campaign_stats(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[CampaignStats]:
    """Get campaign statistics."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign:
        return None
    
    # Get contact counts
    contact_counts = db.query(
        CampaignContact.status,
        func.count(CampaignContact.id).label('count')
    ).filter(
        CampaignContact.campaign_id == campaign_id
    ).group_by(CampaignContact.status).all()
    
    # Get call statistics
    from app.db.models import Call, CallStatus
    call_stats = db.query(
        func.count(Call.id).label('total_calls'),
        func.count(Call.id).filter(Call.status == CallStatus.COMPLETED).label('completed_calls'),
        func.count(Call.id).filter(Call.status == CallStatus.FAILED).label('failed_calls'),
        func.sum(Call.duration_seconds).label('total_duration'),
        func.sum(Call.cost_usd).label('total_cost')
    ).filter(
        Call.campaign_id == campaign_id
    ).first()
    
    # Calculate stats
    stats = {
        'total_contacts': 0,
        'pending_contacts': 0,
        'called_contacts': 0,
        'completed_contacts': 0,
        'failed_contacts': 0,
        'skipped_contacts': 0,
        'total_calls': call_stats.total_calls or 0,
        'completed_calls': call_stats.completed_calls or 0,
        'failed_calls': call_stats.failed_calls or 0,
        'total_duration': call_stats.total_duration or 0,
        'total_cost': float(call_stats.total_cost or 0)
    }
    
    # Map contact status counts
    for status, count in contact_counts:
        if status == CampaignContactStatus.PENDING:
            stats['pending_contacts'] = count
        elif status == CampaignContactStatus.CALLED:
            stats['called_contacts'] = count
        elif status == CampaignContactStatus.COMPLETED:
            stats['completed_contacts'] = count
        elif status == CampaignContactStatus.FAILED:
            stats['failed_contacts'] = count
        elif status == CampaignContactStatus.SKIPPED:
            stats['skipped_contacts'] = count
        stats['total_contacts'] += count
    
    # Calculate success rate
    if stats['total_calls'] > 0:
        stats['success_rate'] = (stats['completed_calls'] / stats['total_calls']) * 100
    else:
        stats['success_rate'] = 0.0
    
    return CampaignStats(**stats)


# Campaign Contact CRUD Operations
def create_campaign_contact(db: Session, contact_data: CampaignContactCreate, campaign_id: uuid.UUID) -> CampaignContact:
    """Create a new campaign contact."""
    contact = CampaignContact(
        campaign_id=campaign_id,
        phone_number=contact_data.phone_number,
        name=contact_data.name,
        email=contact_data.email,
        custom_field_1=contact_data.custom_field_1,
        custom_field_2=contact_data.custom_field_2,
        custom_field_3=contact_data.custom_field_3,
        custom_field_4=contact_data.custom_field_4,
        custom_field_5=contact_data.custom_field_5,
        status=CampaignContactStatus.PENDING
    )
    
    db.add(contact)
    db.commit()
    db.refresh(contact)
    return contact


def get_campaign_contacts(
    db: Session, 
    campaign_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    status: Optional[CampaignContactStatus] = None
) -> List[CampaignContact]:
    """Get contacts for a campaign."""
    query = db.query(CampaignContact).filter(CampaignContact.campaign_id == campaign_id)
    
    if status:
        query = query.filter(CampaignContact.status == status)
    
    return query.order_by(asc(CampaignContact.created_at)).offset(skip).limit(limit).all()


def update_campaign_contact(
    db: Session, 
    contact_id: uuid.UUID, 
    contact_data: CampaignContactUpdate
) -> Optional[CampaignContact]:
    """Update a campaign contact."""
    contact = db.query(CampaignContact).filter(CampaignContact.id == contact_id).first()
    if not contact:
        return None
    
    update_data = contact_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(contact, field, value)
    
    contact.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(contact)
    return contact


def delete_campaign_contact(db: Session, contact_id: uuid.UUID) -> bool:
    """Delete a campaign contact."""
    contact = db.query(CampaignContact).filter(CampaignContact.id == contact_id).first()
    if not contact:
        return False
    
    db.delete(contact)
    db.commit()
    return True
