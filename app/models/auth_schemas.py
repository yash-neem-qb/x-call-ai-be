"""
Pydantic schemas for authentication-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
import uuid


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")


class UserRegister(BaseModel):
    """Schema for user registration."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")
    first_name: Optional[str] = Field(None, max_length=100, description="User first name")
    last_name: Optional[str] = Field(None, max_length=100, description="User last name")
    organization_name: Optional[str] = Field(None, max_length=255, description="Organization name (for new org)")


class Token(BaseModel):
    """Schema for authentication token response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class LoginResponse(BaseModel):
    """Schema for login response with token and user info."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user_id: uuid.UUID = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    first_name: Optional[str] = Field(None, description="User first name")
    last_name: Optional[str] = Field(None, description="User last name")
    full_name: Optional[str] = Field(None, description="User full name")
    organization_id: Optional[uuid.UUID] = Field(None, description="Organization ID")
    organization_name: Optional[str] = Field(None, description="Organization name")


class UserProfile(BaseModel):
    """Schema for user profile response."""
    id: uuid.UUID
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_active: bool
    is_verified: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChangePassword(BaseModel):
    """Schema for changing password."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=6, description="New password")


class ResetPasswordRequest(BaseModel):
    """Schema for password reset request."""
    email: EmailStr = Field(..., description="User email address")


class ResetPassword(BaseModel):
    """Schema for password reset."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=6, description="New password")


class UserOrganization(BaseModel):
    """Schema for user organization information."""
    organization_id: uuid.UUID = Field(..., description="Organization ID")
    organization_name: str = Field(..., description="Organization name")
    role: str = Field(..., description="User role in organization")
    is_active: bool = Field(..., description="Membership status")
    joined_at: datetime = Field(..., description="When user joined the organization")


class UserOrganizationsResponse(BaseModel):
    """Schema for user organizations list response."""
    organizations: List[UserOrganization] = Field(..., description="List of user organizations")
    total_count: int = Field(..., description="Total number of organizations")
