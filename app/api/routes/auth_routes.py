"""
API routes for authentication.
Handles user registration, login, and token management.
"""

import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.auth import (
    verify_password, get_password_hash, create_access_token, 
    get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.db.database import get_db
from app.db.models import User, Organization, OrganizationMember, UserRole, Assistant
from app.models.auth_schemas import (
    UserLogin, UserRegister, Token, LoginResponse, UserProfile, ChangePassword,
    UserOrganizationsResponse, UserOrganization
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/register", response_model=LoginResponse)
async def register_user(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    """Register a new user and create their organization."""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create organization
        org_name = user_data.organization_name or f"{user_data.first_name or 'User'}'s Organization"
        org_slug = org_name.lower().replace(' ', '-').replace('_', '-')
        
        # Ensure unique slug
        counter = 1
        original_slug = org_slug
        while db.query(Organization).filter(Organization.slug == org_slug).first():
            org_slug = f"{original_slug}-{counter}"
            counter += 1
        
        organization = Organization(
            name=org_name,
            slug=org_slug,
            description=f"Organization created by {user_data.email}",
            plan="free",
            max_users=5,
            max_phone_numbers=1,
            max_assistants=3
        )
        
        db.add(organization)
        db.flush()  # Get the organization ID
        
        # Create user
        user = User(
            email=user_data.email,
            password_hash=get_password_hash(user_data.password),
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            organization_id=organization.id,  # Set the organization ID
            role=UserRole.OWNER,  # First user is always owner
            is_active=True,
            is_verified=False,  # Email verification can be added later
            last_login_at=datetime.utcnow()
        )
        
        db.add(user)
        db.flush()  # Get the user ID
        
        # Create organization membership
        membership = OrganizationMember(
            user_id=user.id,
            organization_id=organization.id,
            role=UserRole.OWNER,  # First user is always owner
            is_active=True
        )
        
        db.add(membership)
        
        # Create initial assistant for the user
        initial_assistant = Assistant(
            name="New Assistant",
            organization_id=organization.id,
            team_id=None,  # No team assigned initially
            
            # Voice configuration
            voice_provider="11labs",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            voice_model="eleven_flash_v2_5",
            voice_stability="0.5",
            voice_similarity_boost="0.75",
            voice_speed="1.0",
            
            # LLM configuration
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_system_prompt="You are a customer support specialist who helps users with product questions and issues. Be friendly, empathetic, and solution-oriented.",
            llm_max_tokens=1024,
            llm_temperature="0.7",
            
            # Messages
            first_message="Hello! I'm your customer support specialist. How can I assist you with our products or services today?",
            voicemail_message="Please leave a message and I will get back to you.",
            end_call_message="Thank you for your time. Goodbye.",
            
            # Transcription configuration
            transcriber_provider="deepgram",
            transcriber_model="nova-3",
            transcriber_language="en",
            
            # Security
            is_server_url_secret_set=False,
            
            # RAG configuration
            rag_enabled=True,
            rag_max_results=3,
            rag_score_threshold="0.7",
            rag_max_context_length=2000,
            rag_config={}
        )
        
        db.add(initial_assistant)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "org_id": str(organization.id)},
            expires_delta=access_token_expires
        )
        
        # Commit all changes at once
        db.commit()
        db.refresh(user)
        db.refresh(organization)
        db.refresh(initial_assistant)
        
        logger.info(f"User registered: {user.email} in organization: {organization.id} with initial assistant: {initial_assistant.id}")
        
        # Create full name
        full_name = None
        if user.first_name or user.last_name:
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=full_name,
            organization_id=user.organization_id,
            organization_name=organization.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/login", response_model=LoginResponse)
async def login_user(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token."""
    try:
        # Find user by email
        user = db.query(User).filter(User.email == user_credentials.email).first()
        
        if not user or not verify_password(user_credentials.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Update last login
        user.last_login_at = datetime.utcnow()
        db.commit()
        
        # Get organization information
        organization = None
        organization_name = None
        if user.organization_id:
            organization = db.query(Organization).filter(Organization.id == user.organization_id).first()
            if organization:
                organization_name = organization.name
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "org_id": str(user.organization_id)},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user.email}")
        
        # Create full name
        full_name = None
        if user.first_name or user.last_name:
            full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=full_name,
            organization_id=user.organization_id,
            organization_name=organization_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user's profile information."""
    return current_user


@router.post("/change-password")
async def change_password(
    password_data: ChangePassword,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user's password."""
    try:
        # Verify current password
        if not verify_password(password_data.current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        current_user.password_hash = get_password_hash(password_data.new_password)
        db.commit()
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """Logout user (client should discard token)."""
    logger.info(f"User logged out: {current_user.email}")
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """Refresh access token."""
    try:
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(current_user.id), "email": current_user.email, "org_id": str(current_user.organization_id)},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/organizations", response_model=UserOrganizationsResponse)
async def get_user_organizations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all organizations that the current user is a member of."""
    try:
        # Get all organization memberships for the current user
        memberships = db.query(OrganizationMember).filter(
            OrganizationMember.user_id == current_user.id,
            OrganizationMember.is_active == True
        ).all()
        
        organizations = []
        for membership in memberships:
            # Get organization details
            organization = db.query(Organization).filter(
                Organization.id == membership.organization_id
            ).first()
            
            if organization:
                organizations.append(UserOrganization(
                    organization_id=organization.id,
                    organization_name=organization.name,
                    role=membership.role.value,
                    is_active=membership.is_active,
                    joined_at=membership.joined_at
                ))
        
        logger.info(f"Retrieved {len(organizations)} organizations for user: {current_user.email}")
        
        return UserOrganizationsResponse(
            organizations=organizations,
            total_count=len(organizations)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving user organizations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
