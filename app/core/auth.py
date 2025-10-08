"""
Authentication and authorization utilities.
Handles JWT token creation, validation, and user authentication.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.db.database import get_db
from app.db.models import User, Organization, OrganizationMember, UserRole

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings - using configuration from settings
SECRET_KEY = settings.jwt_secret_key
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_access_token_expire_minutes

# HTTP Bearer token scheme
security = HTTPBearer()


class AuthError(HTTPException):
    """Custom authentication error."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise AuthError("Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user."""
    try:
        # Verify the token
        payload = verify_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthError("Invalid token payload")
        
        # Get user from database
        user = db.query(User).filter(User.id == UUID(user_id)).first()
        if user is None:
            raise AuthError("User not found")
        
        if not user.is_active:
            raise AuthError("Inactive user")
        
        return user
    except ValueError as e:
        logger.error(f"Invalid UUID in token: {e}")
        raise AuthError("Invalid token format")


async def get_current_organization(
    organization_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> tuple[Organization, OrganizationMember]:
    """Get the current user's organization and their membership."""
    # Check if user is a member of the organization
    membership = db.query(OrganizationMember).filter(
        OrganizationMember.user_id == current_user.id,
        OrganizationMember.organization_id == organization_id,
        OrganizationMember.is_active == True
    ).first()
    
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this organization"
        )
    
    organization = membership.organization
    
    if not organization.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization is inactive"
        )
    
    return organization, membership


def require_role_in_org(required_role: str):
    """Decorator to require a specific role in an organization."""
    async def role_checker(
        organization_id: UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ) -> tuple[Organization, OrganizationMember]:
        # Get organization and membership
        organization, membership = await get_current_organization(organization_id, current_user, db)
        
        # Define role hierarchy
        role_hierarchy = {
            "team": 0,
            "admin": 1,
            "owner": 2
        }
        
        user_role_level = role_hierarchy.get(membership.role.value, 0)
        required_role_level = role_hierarchy.get(required_role, 0)
        
        if user_role_level < required_role_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}, your role: {membership.role.value}"
            )
        
        return organization, membership
    
    return role_checker


def require_write_permission():
    """Require admin or owner role (can do POST/PUT/DELETE)."""
    return require_role_in_org("admin")


def require_read_permission():
    """Require any role (can do GET)."""
    async def permission_checker(
        organization_id: UUID,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ) -> tuple[Organization, OrganizationMember]:
        return await get_current_organization(organization_id, current_user, db)
    
    return permission_checker
