"""
CRUD operations for database models.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import Assistant, Call, CallDirection, CallStatus, Campaign, CampaignContact, CampaignStatus, CampaignScheduleType, PhoneNumber

logger = logging.getLogger(__name__)


# Assistant CRUD operations

def create_assistant(db: Session, assistant_data: Dict[str, Any]) -> Assistant:
    """
    Create a new assistant with normalized structure.
    
    Args:
        db: Database session
        assistant_data: Assistant data
        
    Returns:
        Created assistant
    """
    # Extract voice configuration
    voice_config = assistant_data.get("voice", {})
    # Extract LLM configuration
    model_config = assistant_data.get("model", {})
    # Extract transcriber configuration
    transcriber_config = assistant_data.get("transcriber", {})
    
    # Create assistant with normalized columns
    db_assistant = Assistant(
        name=assistant_data.get("name"),
        # Voice configuration
        voice_provider=voice_config.get("provider", "11labs"),
        voice_id=voice_config.get("voiceId"),
        voice_model=voice_config.get("model", "eleven_flash_v2_5"),
        voice_stability=str(voice_config.get("stability", 0.5)),
        voice_similarity_boost=str(voice_config.get("similarityBoost", 0.75)),
        voice_speed=str(voice_config.get("speed", 1.0)),
        # LLM configuration
        llm_provider=model_config.get("provider", "openai"),
        llm_model=model_config.get("model", "gpt-4.1-mini"),
        llm_system_prompt=model_config.get("systemPrompt") or (model_config.get("messages", [{}])[0].get("content") if model_config.get("messages") else None),
        llm_max_tokens=model_config.get("maxTokens"),
        llm_temperature=str(model_config.get("temperature", 0.7)),
        # Messages
        first_message=assistant_data.get("firstMessage"),
        voicemail_message=assistant_data.get("voicemailMessage"),
        end_call_message=assistant_data.get("endCallMessage"),
        # Transcription configuration
        transcriber_provider=transcriber_config.get("provider", "deepgram"),
        transcriber_model=transcriber_config.get("model", "nova-3"),
        transcriber_language=transcriber_config.get("language", "en"),
        # Security
        is_server_url_secret_set=assistant_data.get("isServerUrlSecretSet", False)
    )
    
    db.add(db_assistant)
    db.commit()
    db.refresh(db_assistant)
    logger.info(f"Created assistant: {db_assistant.id} with voice_id: {db_assistant.voice_id}")
    return db_assistant


def get_assistant(db: Session, assistant_id: uuid.UUID) -> Optional[Assistant]:
    """
    Get assistant by ID.
    
    Args:
        db: Database session
        assistant_id: Assistant ID
        
    Returns:
        Assistant or None if not found
    """
    return db.query(Assistant).filter(Assistant.id == assistant_id).first()


def get_assistants(db: Session, skip: int = 0, limit: int = 100) -> List[Assistant]:
    """
    Get all assistants with pagination.
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of assistants
    """
    return db.query(Assistant).offset(skip).limit(limit).all()


def update_assistant(db: Session, assistant_id: uuid.UUID, assistant_data: Dict[str, Any]) -> Optional[Assistant]:
    """
    Update assistant.
    
    Args:
        db: Database session
        assistant_id: Assistant ID
        assistant_data: Assistant data to update
        
    Returns:
        Updated assistant or None if not found
    """
    db_assistant = get_assistant(db, assistant_id)
    if not db_assistant:
        return None
    
    # Update fields
    if "name" in assistant_data:
        db_assistant.name = assistant_data["name"]
    
    # Update voice configuration
    if "voice" in assistant_data:
        voice_config = assistant_data["voice"]
        if "provider" in voice_config:
            db_assistant.voice_provider = voice_config["provider"]
        if "voiceId" in voice_config:
            db_assistant.voice_id = voice_config["voiceId"]
        if "model" in voice_config:
            db_assistant.voice_model = voice_config["model"]
        if "stability" in voice_config:
            db_assistant.voice_stability = str(voice_config["stability"])
        if "similarityBoost" in voice_config:
            db_assistant.voice_similarity_boost = str(voice_config["similarityBoost"])
        if "speed" in voice_config:
            db_assistant.voice_speed = str(voice_config["speed"])
    
    # Update LLM configuration
    if "model" in assistant_data:
        model_config = assistant_data["model"]
        if "provider" in model_config:
            db_assistant.llm_provider = model_config["provider"]
        if "model" in model_config:
            db_assistant.llm_model = model_config["model"]
        if "systemPrompt" in model_config:
            db_assistant.llm_system_prompt = model_config["systemPrompt"]
        if "maxTokens" in model_config:
            db_assistant.llm_max_tokens = model_config["maxTokens"]
        if "temperature" in model_config:
            db_assistant.llm_temperature = str(model_config["temperature"])
    
    # Update messages
    if "firstMessage" in assistant_data:
        db_assistant.first_message = assistant_data["firstMessage"]
    if "voicemailMessage" in assistant_data:
        db_assistant.voicemail_message = assistant_data["voicemailMessage"]
    if "endCallMessage" in assistant_data:
        db_assistant.end_call_message = assistant_data["endCallMessage"]
    
    # Update transcriber configuration
    if "transcriber" in assistant_data:
        transcriber_config = assistant_data["transcriber"]
        if "provider" in transcriber_config:
            db_assistant.transcriber_provider = transcriber_config["provider"]
        if "model" in transcriber_config:
            db_assistant.transcriber_model = transcriber_config["model"]
        if "language" in transcriber_config:
            db_assistant.transcriber_language = transcriber_config["language"]
    
    # Update security
    if "isServerUrlSecretSet" in assistant_data:
        db_assistant.is_server_url_secret_set = assistant_data["isServerUrlSecretSet"]
    
    db.commit()
    db.refresh(db_assistant)
    logger.info(f"Updated assistant: {assistant_id} with voice_id: {db_assistant.voice_id}")
    return db_assistant


def delete_assistant(db: Session, assistant_id: uuid.UUID) -> bool:
    """
    Delete assistant.
    
    Args:
        db: Database session
        assistant_id: Assistant ID
        
    Returns:
        True if assistant was deleted, False otherwise
    """
    db_assistant = get_assistant(db, assistant_id)
    if not db_assistant:
        return False
    
    db.delete(db_assistant)
    db.commit()
    logger.info(f"Deleted assistant: {assistant_id}")
    return True


# Call CRUD operations

def create_call_log(db: Session, call_data: Dict[str, Any]) -> Call:
    """
    Create a new call log record.
    
    Args:
        db: Database session
        call_data: Call data
        
    Returns:
        Created call
    """
    db_call = Call(
        organization_id=call_data.get("organizationId"),
        phone_number_id=call_data.get("phoneNumberId"),
        assistant_id=call_data.get("assistantId"),
        from_number=call_data.get("fromNumber"),
        to_number=call_data.get("toNumber"),
        direction=CallDirection.WEB if call_data.get("direction") == "web" else CallDirection(call_data.get("direction", "web")),
        status=CallStatus.INITIATED if call_data.get("status") == "initiated" else CallStatus(call_data.get("status", "initiated")),
        session_id=call_data.get("sessionId"),
        twilio_call_sid=call_data.get("twilioCallSid"),
        twilio_account_sid=call_data.get("twilioAccountSid"),
        started_at=call_data.get("startedAt"),
        ended_at=call_data.get("endedAt"),
        duration_seconds=call_data.get("durationSeconds"),
        cost_usd=call_data.get("costUsd"),
        created_by=call_data.get("createdBy")
    )
    
    db.add(db_call)
    db.commit()
    db.refresh(db_call)
    logger.info(f"Created call log: {db_call.id}")
    return db_call


def get_call_log(db: Session, call_id: uuid.UUID) -> Optional[Call]:
    """
    Get call log by ID.
    
    Args:
        db: Database session
        call_id: Call ID
        
    Returns:
        Call or None if not found
    """
    return db.query(Call).filter(Call.id == call_id).first()


def get_call_by_twilio_sid(db: Session, twilio_call_sid: str) -> Optional[Call]:
    """
    Get a call log by Twilio Call SID.
    
    Args:
        db: Database session
        twilio_call_sid: Twilio Call SID
        
    Returns:
        Call log or None if not found
    """
    return db.query(Call).filter(Call.twilio_call_sid == twilio_call_sid).first()


def get_call_by_session_id(db: Session, session_id: str) -> Optional[Call]:
    """
    Get a call log by session ID (for web calls).
    
    Args:
        db: Database session
        session_id: Session ID
        
    Returns:
        Call log or None if not found
    """
    return db.query(Call).filter(Call.session_id == session_id).first()


def update_call_log(db: Session, call_id: uuid.UUID, update_data: Dict[str, Any]) -> Optional[Call]:
    """
    Update a call log record.
    
    Args:
        db: Database session
        call_id: Call ID
        update_data: Data to update
        
    Returns:
        Updated call or None if not found
    """
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        return None
    
    # Update fields
    for key, value in update_data.items():
        if hasattr(call, key) and value is not None:
            setattr(call, key, value)
    
    db.commit()
    db.refresh(call)
    logger.info(f"Updated call log: {call.id}")
    return call


def update_call_by_twilio_sid(db: Session, twilio_call_sid: str, update_data: Dict[str, Any]) -> Optional[Call]:
    """
    Update a call log record by Twilio Call SID.
    
    Args:
        db: Database session
        twilio_call_sid: Twilio Call SID
        update_data: Data to update
        
    Returns:
        Updated call or None if not found
    """
    call = db.query(Call).filter(Call.twilio_call_sid == twilio_call_sid).first()
    if not call:
        return None
    
    # Update fields
    for key, value in update_data.items():
        if hasattr(call, key) and value is not None:
            setattr(call, key, value)
    
    db.commit()
    db.refresh(call)
    logger.info(f"Updated call log by Twilio SID: {twilio_call_sid}")
    return call


def add_transcript_message(db: Session, call_id: uuid.UUID, message_data: Dict[str, Any]) -> Optional[Call]:
    """
    Add a transcript message to a call.
    
    Args:
        db: Database session
        call_id: Call ID
        message_data: Message data with speaker, message, timestamp, etc.
        
    Returns:
        Updated call or None if not found
    """
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        return None
    
    # Initialize transcript_data if it doesn't exist
    if call.transcript_data is None:
        call.transcript_data = []
    
    # Add the new message
    call.transcript_data.append(message_data)
    
    # Mark the field as modified so SQLAlchemy knows to update it
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(call, 'transcript_data')
    
    db.commit()
    db.refresh(call)
    logger.info(f"Added transcript message to call: {call.id}")
    return call


def add_transcript_message_by_twilio_sid(db: Session, twilio_call_sid: str, message_data: Dict[str, Any]) -> Optional[Call]:
    """
    Add a transcript message to a call by Twilio Call SID.
    
    Args:
        db: Database session
        twilio_call_sid: Twilio Call SID
        message_data: Message data with speaker, message, timestamp, etc.
        
    Returns:
        Updated call or None if not found
    """
    call = db.query(Call).filter(Call.twilio_call_sid == twilio_call_sid).first()
    if not call:
        return None
    
    # Initialize transcript_data if it doesn't exist
    if call.transcript_data is None:
        call.transcript_data = []
    
    # Add the new message
    call.transcript_data.append(message_data)
    
    # Mark the field as modified so SQLAlchemy knows to update it
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(call, 'transcript_data')
    
    db.commit()
    db.refresh(call)
    logger.info(f"Added transcript message to call by Twilio SID: {twilio_call_sid}")
    return call


def get_call_logs(db: Session, organization_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[Call]:
    """
    Get call logs for an organization with pagination.
    
    Args:
        db: Database session
        organization_id: Organization ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of calls
    """
    return db.query(Call).filter(Call.organization_id == organization_id).offset(skip).limit(limit).all()


def update_call_log(db: Session, call_id: uuid.UUID, call_data: Dict[str, Any]) -> Optional[Call]:
    """
    Update call log.
    
    Args:
        db: Database session
        call_id: Call ID
        call_data: Call data to update
        
    Returns:
        Updated call or None if not found
    """
    db_call = get_call_log(db, call_id)
    if not db_call:
        return None
    
    # Update fields
    if "status" in call_data:
        db_call.status = CallStatus(call_data["status"])
    if "endedAt" in call_data:
        db_call.ended_at = call_data["endedAt"]
    if "durationSeconds" in call_data:
        db_call.duration_seconds = call_data["durationSeconds"]
    if "costUsd" in call_data:
        db_call.cost_usd = call_data["costUsd"]
    if "transcription" in call_data:
        db_call.transcription = call_data["transcription"]
    if "recordingUrl" in call_data:
        db_call.recording_url = call_data["recordingUrl"]
    
    db.commit()
    db.refresh(db_call)
    logger.info(f"Updated call log: {call_id}")
    return db_call


def delete_call_log(db: Session, call_id: uuid.UUID) -> bool:
    """
    Delete call log.
    
    Args:
        db: Database session
        call_id: Call ID
        
    Returns:
        True if call was deleted, False otherwise
    """
    db_call = get_call_log(db, call_id)
    if not db_call:
        return False
    
    db.delete(db_call)
    db.commit()
    logger.info(f"Deleted call log: {call_id}")
    return True


# Campaign CRUD operations

def create_campaign(db: Session, campaign_data, organization_id: uuid.UUID):
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
        if hasattr(campaign_data, 'contacts') and campaign_data.contacts:
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
                    status="pending"
                )
                db.add(contact)
        
        db.commit()
        db.refresh(campaign)
        return campaign
        
    except Exception as e:
        db.rollback()
        raise e


def get_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID):
    """Get a campaign by ID."""
    return db.query(Campaign).filter(
        Campaign.id == campaign_id,
        Campaign.organization_id == organization_id
    ).first()


def get_campaigns(
    db: Session, 
    organization_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    search: Optional[str] = None
):
    """Get campaigns for an organization with pagination and filtering."""
    query = db.query(Campaign).filter(Campaign.organization_id == organization_id)
    
    # Apply filters
    if status:
        query = query.filter(Campaign.status == status)
    
    if search:
        query = query.filter(
            Campaign.name.ilike(f"%{search}%")
        )
    
    return query.order_by(Campaign.created_at.desc()).offset(skip).limit(limit).all()


def get_campaigns_count(
    db: Session, 
    organization_id: uuid.UUID,
    status: Optional[str] = None,
    search: Optional[str] = None
) -> int:
    """Get total count of campaigns for an organization."""
    query = db.query(Campaign).filter(Campaign.organization_id == organization_id)
    
    if status:
        query = query.filter(Campaign.status == status)
    
    if search:
        query = query.filter(
            Campaign.name.ilike(f"%{search}%")
        )
    
    return query.count()


def update_campaign(
    db: Session, 
    campaign_id: uuid.UUID, 
    organization_id: uuid.UUID,
    campaign_data
):
    """Update a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign:
        return None
    
    update_data = campaign_data.dict(exclude_unset=True) if hasattr(campaign_data, 'dict') else campaign_data
    for field, value in update_data.items():
        if hasattr(campaign, field):
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


def start_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID):
    """Start a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status != "draft":
        return None
    
    campaign.status = "active"
    campaign.started_at = datetime.utcnow()
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def pause_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID):
    """Pause a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status != "active":
        return None
    
    campaign.status = "paused"
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def resume_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID):
    """Resume a paused campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status != "paused":
        return None
    
    campaign.status = "active"
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def stop_campaign(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID):
    """Stop a campaign."""
    campaign = get_campaign(db, campaign_id, organization_id)
    if not campaign or campaign.status not in ["active", "paused"]:
        return None
    
    campaign.status = "completed"
    campaign.completed_at = datetime.utcnow()
    campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(campaign)
    return campaign


def get_campaign_stats(db: Session, campaign_id: uuid.UUID, organization_id: uuid.UUID):
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
        if status == "pending":
            stats['pending_contacts'] = count
        elif status == "called":
            stats['called_contacts'] = count
        elif status == "completed":
            stats['completed_contacts'] = count
        elif status == "failed":
            stats['failed_contacts'] = count
        elif status == "skipped":
            stats['skipped_contacts'] = count
        stats['total_contacts'] += count
    
    # Calculate success rate
    if stats['total_calls'] > 0:
        stats['success_rate'] = (stats['completed_calls'] / stats['total_calls']) * 100
    else:
        stats['success_rate'] = 0.0
    
    return stats


def create_campaign_contact(db: Session, contact_data, campaign_id: uuid.UUID):
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
        status="pending"
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
    status: Optional[str] = None
):
    """Get contacts for a campaign."""
    query = db.query(CampaignContact).filter(CampaignContact.campaign_id == campaign_id)
    
    if status:
        query = query.filter(CampaignContact.status == status)
    
    return query.order_by(CampaignContact.created_at.asc()).offset(skip).limit(limit).all()


def update_campaign_contact(
    db: Session, 
    contact_id: uuid.UUID, 
    contact_data
):
    """Update a campaign contact."""
    contact = db.query(CampaignContact).filter(CampaignContact.id == contact_id).first()
    if not contact:
        return None
    
    update_data = contact_data.dict(exclude_unset=True) if hasattr(contact_data, 'dict') else contact_data
    for field, value in update_data.items():
        if hasattr(contact, field):
            setattr(contact, field, value)
    
    contact.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(contact)
    return contact


# Phone Number CRUD operations

def create_phone_number(db: Session, phone_data: Dict[str, Any]) -> PhoneNumber:
    """
    Create a new phone number.
    
    Args:
        db: Database session
        phone_data: Phone number data
        
    Returns:
        Created phone number
    """
    db_phone_number = PhoneNumber(
        phone_number=phone_data.get("phone_number"),
        organization_id=phone_data.get("organization_id"),
        assistant_id=phone_data.get("assistant_id"),
        twilio_account_sid=phone_data.get("twilio_account_sid"),
        twilio_auth_token=phone_data.get("twilio_auth_token"),
        twilio_phone_number_sid=phone_data.get("twilio_phone_number_sid"),
        voice_webhook_url=phone_data.get("voice_webhook_url"),
        status_callback_url=phone_data.get("status_callback_url"),
        friendly_name=phone_data.get("friendly_name"),
        settings=phone_data.get("settings", {}),
        is_active=phone_data.get("is_active", True),
        is_configured=phone_data.get("is_configured", False)
    )
    
    db.add(db_phone_number)
    db.commit()
    db.refresh(db_phone_number)
    logger.info(f"Created phone number: {db_phone_number.id}")
    return db_phone_number


def get_phone_number(db: Session, phone_number_id: uuid.UUID) -> Optional[PhoneNumber]:
    """
    Get a phone number by ID.
    
    Args:
        db: Database session
        phone_number_id: Phone number ID
        
    Returns:
        Phone number or None
    """
    return db.query(PhoneNumber).filter(PhoneNumber.id == phone_number_id).first()


def get_phone_numbers(db: Session, organization_id: uuid.UUID, skip: int = 0, limit: int = 100) -> List[PhoneNumber]:
    """
    Get phone numbers for an organization.
    
    Args:
        db: Database session
        organization_id: Organization ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of phone numbers
    """
    return (
        db.query(PhoneNumber)
        .filter(PhoneNumber.organization_id == organization_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_phone_number(db: Session, phone_number_id: uuid.UUID, phone_data: Dict[str, Any]) -> Optional[PhoneNumber]:
    """
    Update a phone number.
    
    Args:
        db: Database session
        phone_number_id: Phone number ID
        phone_data: Phone number data to update
        
    Returns:
        Updated phone number or None
    """
    phone_number = db.query(PhoneNumber).filter(PhoneNumber.id == phone_number_id).first()
    if not phone_number:
        return None
    
    for key, value in phone_data.items():
        if hasattr(phone_number, key):
            setattr(phone_number, key, value)
    
    phone_number.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(phone_number)
    logger.info(f"Updated phone number: {phone_number_id}")
    return phone_number


def delete_phone_number(db: Session, phone_number_id: uuid.UUID) -> bool:
    """
    Delete a phone number.
    
    Args:
        db: Database session
        phone_number_id: Phone number ID
        
    Returns:
        True if deleted, False if not found
    """
    phone_number = db.query(PhoneNumber).filter(PhoneNumber.id == phone_number_id).first()
    if not phone_number:
        return False
    
    db.delete(phone_number)
    db.commit()
    logger.info(f"Deleted phone number: {phone_number_id}")
    return True