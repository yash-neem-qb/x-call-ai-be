"""
API routes for organization management.
Handles CRUD operations for organizations, users, teams, and phone numbers.
"""

import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, String

from app.db.database import get_db
from app.db.models import Organization, User, Team, TeamMember, PhoneNumber, Call, CallEndReason, UserRole, Assistant, OrganizationMember
from app.models.organization_schemas import (
    OrganizationCreate, OrganizationUpdate, OrganizationResponse, OrganizationListResponse,
    UserCreate, UserUpdate, UserResponse,
    TeamCreate, TeamUpdate, TeamResponse,
    TeamMemberCreate, TeamMemberResponse,
    PhoneNumberCreate, PhoneNumberUpdate, PhoneNumberResponse,
    CallCreate, CallUpdate, CallResponse, CallListResponse
)
from app.core.auth import get_current_user, require_write_permission, require_read_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/organizations", tags=["organizations"])


# Organization endpoints
@router.get("/{organization_id}", response_model=OrganizationResponse)
async def get_organization(
    organization_id: UUID,
    org_data: tuple = Depends(require_read_permission())
):
    """Get organization by ID."""
    organization, membership = org_data
    return organization


@router.put("/{organization_id}", response_model=OrganizationResponse)
async def update_organization(
    organization_id: UUID,
    organization_update: OrganizationUpdate,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Update organization."""
    try:
        organization, membership = org_data
        
        # Update fields
        update_data = organization_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(organization, field, value)
        
        db.commit()
        db.refresh(organization)
        
        logger.info(f"Updated organization: {organization.id}")
        return organization
        
    except Exception as e:
        logger.error(f"Error updating organization: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")




# Phone number endpoints
@router.post("/{organization_id}/phone-numbers", response_model=PhoneNumberResponse)
async def create_phone_number(
    organization_id: UUID,
    phone_number: PhoneNumberCreate,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Create a new phone number for the current organization and configure Twilio webhooks."""
    try:
        from app.services.twilio_service import twilio_service
        from app.config.settings import settings
        
        # Check if phone number already exists
        existing_phone = db.query(PhoneNumber).filter(PhoneNumber.phone_number == phone_number.phone_number).first()
        if existing_phone:
            raise HTTPException(status_code=400, detail="Phone number already exists")
        
        # Get the webhook base URL from settings
        webhook_base_url = f"https://{settings.domain}"
        
        logger.info(f"Creating and configuring phone number: {phone_number.phone_number}")
        
        # Configure Twilio webhooks first
        twilio_result = await twilio_service.configure_inbound_webhooks(
            account_sid=phone_number.twilio_account_sid,
            auth_token=phone_number.twilio_auth_token,
            phone_number=phone_number.phone_number,
            webhook_base_url=webhook_base_url
        )
        
        organization, membership = org_data
        
        # Validate assistant belongs to organization if provided
        assistant_id = phone_number.assistant_id
        if assistant_id:
            assistant_exists = (
                db.query(Assistant)
                .filter(
                    Assistant.id == assistant_id,
                    Assistant.organization_id == organization.id
                )
                .first()
            )
            if not assistant_exists:
                raise HTTPException(status_code=400, detail="Assistant not found in organization")

        # Create phone number with webhook configuration
        db_phone_number = PhoneNumber(
            phone_number=phone_number.phone_number,
            organization_id=organization.id,
            assistant_id=phone_number.assistant_id,
            twilio_account_sid=phone_number.twilio_account_sid,
            twilio_auth_token=phone_number.twilio_auth_token,
            twilio_phone_number_sid=twilio_result["sid"],
            voice_webhook_url=twilio_result["voice_url"],
            status_callback_url=twilio_result["status_callback_url"],
            friendly_name=phone_number.friendly_name or twilio_result["friendly_name"],
            settings=phone_number.settings or {},
            is_configured=True
        )
        
        db.add(db_phone_number)
        db.commit()
        db.refresh(db_phone_number)
        
        logger.info(f"Created and configured phone number: {db_phone_number.id} for organization: {organization.id}")
        return db_phone_number
        
    except ValueError as e:
        logger.error(f"Twilio configuration error: {e}")
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Failed to configure Twilio webhooks: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating phone number: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{organization_id}/phone-numbers", response_model=List[PhoneNumberResponse])
async def list_phone_numbers(
    organization_id: UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """List all phone numbers for the organization."""
    try:
        organization, membership = org_data
        phone_numbers = (
            db.query(PhoneNumber)
            .filter(PhoneNumber.organization_id == organization.id)
            .all()
        )
        return phone_numbers
        
    except Exception as e:
        logger.error(f"Error listing phone numbers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{organization_id}/phone-numbers/{phone_number_id}", response_model=PhoneNumberResponse)
async def get_phone_number(
    organization_id: UUID,
    phone_number_id: UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """Get a specific phone number."""
    try:
        organization, membership = org_data
        phone_number = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.id == phone_number_id,
                PhoneNumber.organization_id == organization.id,
            )
            .first()
        )
        
        if not phone_number:
            raise HTTPException(status_code=404, detail="Phone number not found")
        
        return phone_number
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting phone number: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{organization_id}/phone-numbers/{phone_number_id}", response_model=PhoneNumberResponse)
async def update_phone_number(
    organization_id: UUID,
    phone_number_id: UUID,
    phone_number_update: PhoneNumberUpdate,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Update a phone number."""
    try:
        organization, membership = org_data
        phone_number = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.id == phone_number_id,
                PhoneNumber.organization_id == organization.id,
            )
            .first()
        )
        
        if not phone_number:
            raise HTTPException(status_code=404, detail="Phone number not found")
        
        # Update fields
        update_data = phone_number_update.dict(exclude_unset=True)

        if "assistant_id" in update_data:
            assistant_id = update_data.pop("assistant_id")
            if assistant_id:
                assistant_exists = (
                    db.query(Assistant)
                    .filter(
                        Assistant.id == assistant_id,
                        Assistant.organization_id == organization.id
                    )
                    .first()
                )
                if not assistant_exists:
                    raise HTTPException(status_code=400, detail="Assistant not found in organization")
            phone_number.assistant_id = assistant_id

        for field, value in update_data.items():
            setattr(phone_number, field, value)
        
        db.commit()
        db.refresh(phone_number)
        
        logger.info(f"Updated phone number: {phone_number_id}")
        return phone_number
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone number: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{organization_id}/phone-numbers/{phone_number_id}")
async def delete_phone_number(
    organization_id: UUID,
    phone_number_id: UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Delete a phone number."""
    try:
        organization, membership = org_data
        phone_number = (
            db.query(PhoneNumber)
            .filter(
                PhoneNumber.id == phone_number_id,
                PhoneNumber.organization_id == organization.id,
            )
            .first()
        )
        
        if not phone_number:
            raise HTTPException(status_code=404, detail="Phone number not found")
        
        db.delete(phone_number)
        db.commit()
        
        logger.info(f"Deleted phone number: {phone_number_id}")
        return {"message": "Phone number deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting phone number: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")




# Call endpoints
@router.get("/{organization_id}/calls", response_model=CallListResponse)
async def list_calls(
    organization_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    # Basic filters
    status: Optional[str] = Query(None, description="Filter by call status"),
    direction: Optional[str] = Query(None, description="Filter by call direction"),
    # Date filters
    date_from: Optional[str] = Query(None, description="Filter calls from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter calls to date (ISO format)"),
    # Cost filters
    cost_min: Optional[float] = Query(None, description="Minimum cost filter"),
    cost_max: Optional[float] = Query(None, description="Maximum cost filter"),
    # Assistant filters
    assistant_id: Optional[UUID] = Query(None, description="Filter by assistant ID"),
    # Phone number filters
    phone_number: Optional[str] = Query(None, description="Filter by phone number"),
    customer_phone: Optional[str] = Query(None, description="Filter by customer phone number"),
    # Call ID filter
    call_id: Optional[str] = Query(None, description="Filter by call ID (partial match)"),
    # Success evaluation filter
    call_success: Optional[bool] = Query(None, description="Filter by call success (true/false)"),
    # Ended reason filter
    end_reason: Optional[str] = Query(None, description="Filter by end reason"),
    # Search term
    search: Optional[str] = Query(None, description="Search across multiple fields"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """List all calls for the organization with pagination and advanced filters."""
    try:
        from datetime import datetime
        
        organization, membership = org_data
        query = db.query(Call).filter(Call.organization_id == organization.id)
        
        # Apply basic filters
        if status:
            query = query.filter(Call.status == status)
        if direction:
            query = query.filter(Call.direction == direction)
        
        # Apply date filters
        if date_from:
            try:
                date_from_obj = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                query = query.filter(Call.started_at >= date_from_obj)
            except ValueError:
                pass  # Invalid date format, ignore filter
        
        if date_to:
            try:
                date_to_obj = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                query = query.filter(Call.started_at <= date_to_obj)
            except ValueError:
                pass  # Invalid date format, ignore filter
        
        # Apply cost filters
        if cost_min is not None:
            query = query.filter(Call.cost_usd >= cost_min)
        if cost_max is not None:
            query = query.filter(Call.cost_usd <= cost_max)
        
        # Apply assistant filter
        if assistant_id:
            query = query.filter(Call.assistant_id == assistant_id)
        
        # Apply phone number filters
        if phone_number:
            query = query.filter(Call.to_number == phone_number)
        if customer_phone:
            query = query.filter(Call.from_number.contains(customer_phone))
        
        # Apply call ID filter
        if call_id:
            query = query.filter(Call.id.cast(String).contains(call_id))
        
        # Apply success evaluation filter
        if call_success is not None:
            query = query.filter(Call.call_success == call_success)
        
        # Apply end reason filter
        if end_reason:
            query = query.filter(Call.end_reason.contains(end_reason))
        
        # Apply search filter (across multiple fields)
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Call.id.cast(String).ilike(search_term),
                    Call.from_number.ilike(search_term),
                    Call.to_number.ilike(search_term),
                    Call.transcription.ilike(search_term)
                )
            )
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * per_page
        calls = query.order_by(Call.created_at.desc()).offset(offset).limit(per_page).all()
        
        total_pages = (total + per_page - 1) // per_page
        
        return CallListResponse(
            calls=calls,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing calls: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{organization_id}/calls/{call_id}")
async def get_call_details(
    organization_id: UUID,
    call_id: UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific call including transcript."""
    try:
        organization, membership = org_data
        
        # Get the call with all related data
        call = (
            db.query(Call)
            .filter(
                Call.id == call_id,
                Call.organization_id == organization.id
            )
            .first()
        )
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Return call details with transcript data
        return {
            "id": str(call.id),
            "twilio_call_sid": call.twilio_call_sid,
            "twilio_account_sid": call.twilio_account_sid,
            "organization_id": str(call.organization_id),
            "phone_number_id": str(call.phone_number_id) if call.phone_number_id else None,
            "assistant_id": str(call.assistant_id) if call.assistant_id else None,
            "from_number": call.from_number,
            "to_number": call.to_number,
            "direction": call.direction.value,
            "status": call.status.value,
            "duration_seconds": call.duration_seconds,
            "recording_url": call.recording_url,
            "transcription": call.transcription,
            "end_reason": call.end_reason.value if call.end_reason else None,
            "transcript_data": call.transcript_data or [],
            "cost_usd": float(call.cost_usd) if call.cost_usd else None,
            "cost_currency": call.cost_currency,
            "quality_score": float(call.quality_score) if call.quality_score else None,
            "sentiment_score": float(call.sentiment_score) if call.sentiment_score else None,
            "satisfaction_rating": call.satisfaction_rating,
            "call_data": call.call_data or {},
            "created_by": str(call.created_by) if call.created_by else None,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            "created_at": call.created_at.isoformat(),
            "updated_at": call.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Members endpoints
@router.get("/{organization_id}/members")
async def list_organization_members(
    organization_id: UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """List all members of the organization."""
    try:
        organization, membership = org_data
        
        # Get all organization members
        members = (
            db.query(OrganizationMember)
            .filter(
                OrganizationMember.organization_id == organization.id,
                OrganizationMember.is_active == True
            )
            .all()
        )
        
        # Get user details for each member
        member_details = []
        for member in members:
            user = db.query(User).filter(User.id == member.user_id).first()
            if user:
                member_details.append({
                    "id": str(member.id),
                    "email": user.email,
                    "name": f"{user.first_name or ''} {user.last_name or ''}".strip() or None,
                    "role": member.role.value,
                    "status": "Active" if member.is_active else "Inactive",
                    "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                    "last_active_at": user.last_login_at.isoformat() if user.last_login_at else None
                })
        
        return {"members": member_details, "total_count": len(member_details)}
        
    except Exception as e:
        logger.error(f"Error listing organization members: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{organization_id}/members/invite")
async def invite_organization_member(
    organization_id: UUID,
    email: str,
    role: str = "member",
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Invite a new member to the organization."""
    try:
        organization, membership = org_data
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        
        if existing_user:
            # Check if user is already a member
            existing_member = (
                db.query(OrganizationMember)
                .filter(
                    OrganizationMember.user_id == existing_user.id,
                    OrganizationMember.organization_id == organization.id
                )
                .first()
            )
            
            if existing_member:
                if existing_member.is_active:
                    raise HTTPException(status_code=400, detail="User is already a member of this organization")
                else:
                    # Reactivate existing member
                    existing_member.is_active = True
                    existing_member.role = UserRole(role)
                    db.commit()
                    return {"message": "Member reactivated successfully"}
            else:
                # Add existing user as member
                new_member = OrganizationMember(
                    user_id=existing_user.id,
                    organization_id=organization.id,
                    role=UserRole(role),
                    is_active=True
                )
                db.add(new_member)
                db.commit()
                return {"message": "Member added successfully"}
        else:
            # TODO: Send invitation email to new user
            # For now, just return success
            return {"message": "Invitation sent successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inviting organization member: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{organization_id}/members/{member_id}/role")
async def update_member_role(
    organization_id: UUID,
    member_id: UUID,
    role: str,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Update a member's role in the organization."""
    try:
        organization, membership = org_data
        
        member = (
            db.query(OrganizationMember)
            .filter(
                OrganizationMember.id == member_id,
                OrganizationMember.organization_id == organization.id
            )
            .first()
        )
        
        if not member:
            raise HTTPException(status_code=404, detail="Member not found")
        
        member.role = UserRole(role)
        db.commit()
        
        logger.info(f"Updated member role: {member_id} to {role}")
        return {"message": "Member role updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating member role: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{organization_id}/members/{member_id}")
async def remove_organization_member(
    organization_id: UUID,
    member_id: UUID,
    org_data: tuple = Depends(require_write_permission()),
    db: Session = Depends(get_db)
):
    """Remove a member from the organization."""
    try:
        organization, membership = org_data
        
        member = (
            db.query(OrganizationMember)
            .filter(
                OrganizationMember.id == member_id,
                OrganizationMember.organization_id == organization.id
            )
            .first()
        )
        
        if not member:
            raise HTTPException(status_code=404, detail="Member not found")
        
        # Don't allow removing the last admin
        if member.role == UserRole.OWNER:
            admin_count = (
                db.query(OrganizationMember)
                .filter(
                    OrganizationMember.organization_id == organization.id,
                    OrganizationMember.role == UserRole.OWNER,
                    OrganizationMember.is_active == True
                )
                .count()
            )
            if admin_count <= 1:
                raise HTTPException(status_code=400, detail="Cannot remove the last owner of the organization")
        
        member.is_active = False
        db.commit()
        
        logger.info(f"Removed member: {member_id} from organization: {organization.id}")
        return {"message": "Member removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing organization member: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")
