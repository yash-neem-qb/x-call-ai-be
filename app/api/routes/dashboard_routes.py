"""
API routes for dashboard analytics and reporting.
"""

import uuid
import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.core.auth import require_read_permission
from app.services.analytics_service import analytics_service

logger = logging.getLogger(__name__)


def calculate_trends(current_data: dict, previous_data: dict = None) -> dict:
    """
    Calculate trend percentages between current and previous period data.
    
    Args:
        current_data: Current period dashboard data
        previous_data: Previous period dashboard data
        
    Returns:
        Dictionary with trend calculations
    """
    if not previous_data:
        return {
            "call_trend": 0.0,
            "cost_trend": 0.0,
            "success_trend": 0.0,
            "duration_trend": 0.0
        }
    
    trends = {}
    
    # Calculate call trend
    current_calls = current_data.get("overview", {}).get("total_calls", 0)
    previous_calls = previous_data.get("overview", {}).get("total_calls", 0)
    if previous_calls > 0:
        trends["call_trend"] = float((current_calls - previous_calls) / previous_calls) * 100
    else:
        trends["call_trend"] = 100.0 if current_calls > 0 else 0.0
    
    # Calculate cost trend
    current_cost = current_data.get("cost_metrics", {}).get("total_cost_usd", 0)
    previous_cost = previous_data.get("cost_metrics", {}).get("total_cost_usd", 0)
    if previous_cost > 0:
        trends["cost_trend"] = float((current_cost - previous_cost) / previous_cost) * 100
    else:
        trends["cost_trend"] = 100.0 if current_cost > 0 else 0.0
    
    # Calculate success rate trend
    current_success = current_data.get("overview", {}).get("success_rate", 0)
    previous_success = previous_data.get("overview", {}).get("success_rate", 0)
    trends["success_trend"] = current_success - previous_success
    
    # Calculate duration trend
    current_duration = current_data.get("duration_metrics", {}).get("average_duration_seconds", 0)
    previous_duration = previous_data.get("duration_metrics", {}).get("average_duration_seconds", 0)
    if previous_duration > 0:
        trends["duration_trend"] = float((current_duration - previous_duration) / previous_duration) * 100
    else:
        trends["duration_trend"] = 100.0 if current_duration > 0 else 0.0
    
    return trends

router = APIRouter(prefix="/api/v1/organizations", tags=["dashboard"])


@router.get("/{organization_id}/dashboard/overview")
async def get_dashboard_overview(
    organization_id: uuid.UUID,
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    assistant_id: Optional[uuid.UUID] = Query(None, description="Filter by specific assistant ID"),
    period_days: Optional[int] = Query(None, description="Number of days to look back (overrides date_from/date_to)"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive dashboard overview metrics with filtering options.
    
    Args:
        organization_id: Organization ID
        date_from: Optional start date filter (YYYY-MM-DD)
        date_to: Optional end date filter (YYYY-MM-DD)
        assistant_id: Optional filter by specific assistant ID
        period_days: Optional number of days to look back (overrides date_from/date_to)
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Dashboard overview metrics with trends and comparisons
    """
    try:
        organization, membership = org_data
        
        # Parse dates - period_days takes precedence over date_from/date_to
        parsed_date_from = None
        parsed_date_to = None
        
        if period_days:
            # Use period_days to calculate date range
            parsed_date_to = datetime.utcnow()
            parsed_date_from = parsed_date_to - timedelta(days=period_days)
        else:
            # Use provided date_from/date_to
            if date_from:
                try:
                    parsed_date_from = datetime.strptime(date_from, "%Y-%m-%d")
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
            
            if date_to:
                try:
                    parsed_date_to = datetime.strptime(date_to, "%Y-%m-%d")
                    # Set to end of day
                    parsed_date_to = parsed_date_to.replace(hour=23, minute=59, second=59)
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        # Get dashboard overview with filters
        overview = await analytics_service.get_dashboard_overview(
            db=db,
            organization_id=str(organization.id),
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            assistant_id=str(assistant_id) if assistant_id else None
        )
        
        # Get previous period data for trend calculation
        previous_period_overview = None
        if parsed_date_from and parsed_date_to:
            period_duration = parsed_date_to - parsed_date_from
            prev_date_to = parsed_date_from
            prev_date_from = prev_date_to - period_duration
            
            previous_period_overview = await analytics_service.get_dashboard_overview(
                db=db,
                organization_id=str(organization.id),
                date_from=prev_date_from,
                date_to=prev_date_to,
                assistant_id=str(assistant_id) if assistant_id else None
            )
        
        # Calculate trends
        trends = calculate_trends(overview, previous_period_overview)
        
        return {
            "overview": overview,
            "trends": trends,
            "filters": {
                "assistant_id": str(assistant_id) if assistant_id else None,
                "date_from": parsed_date_from.isoformat() if parsed_date_from else None,
                "date_to": parsed_date_to.isoformat() if parsed_date_to else None,
                "period_days": period_days
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard overview")


@router.get("/{organization_id}/dashboard/trends")
async def get_call_trends(
    organization_id: uuid.UUID,
    period: str = Query("day", description="Aggregation period (day, week, month)"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get call trends over time.
    
    Args:
        organization_id: Organization ID
        period: Aggregation period (day, week, month)
        days: Number of days to look back
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Call trends data
    """
    try:
        organization, membership = org_data
        
        # Validate period
        if period not in ["day", "week", "month"]:
            raise HTTPException(status_code=400, detail="Period must be one of: day, week, month")
        
        # Get call trends
        trends = await analytics_service.get_call_trends(
            db=db,
            organization_id=str(organization.id),
            period=period,
            days=days
        )
        
        return trends
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to get call trends")


@router.get("/{organization_id}/dashboard/phone-numbers")
async def get_phone_number_analytics(
    organization_id: uuid.UUID,
    phone_number_id: Optional[uuid.UUID] = Query(None, description="Specific phone number ID"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get detailed phone number analytics.
    
    Args:
        organization_id: Organization ID
        phone_number_id: Optional specific phone number ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Phone number analytics
    """
    try:
        organization, membership = org_data
        
        # Get phone number analytics
        analytics = await analytics_service.get_phone_number_analytics(
            db=db,
            organization_id=str(organization.id),
            phone_number_id=str(phone_number_id) if phone_number_id else None
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting phone number analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get phone number analytics")


@router.get("/{organization_id}/dashboard/assistants")
async def get_assistant_analytics(
    organization_id: uuid.UUID,
    assistant_id: Optional[uuid.UUID] = Query(None, description="Specific assistant ID"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get detailed assistant analytics.
    
    Args:
        organization_id: Organization ID
        assistant_id: Optional specific assistant ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Assistant analytics
    """
    try:
        organization, membership = org_data
        
        # Get assistant analytics
        analytics = await analytics_service.get_assistant_analytics(
            db=db,
            organization_id=str(organization.id),
            assistant_id=str(assistant_id) if assistant_id else None
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting assistant analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get assistant analytics")


@router.get("/{organization_id}/dashboard/hourly-distribution")
async def get_hourly_distribution(
    organization_id: uuid.UUID,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get call distribution by hour of day.
    
    Args:
        organization_id: Organization ID
        days: Number of days to analyze
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Hourly distribution data
    """
    try:
        organization, membership = org_data
        
        # Get hourly distribution
        distribution = await analytics_service.get_hourly_distribution(
            db=db,
            organization_id=str(organization.id),
            days=days
        )
        
        return distribution
        
    except Exception as e:
        logger.error(f"Error getting hourly distribution: {e}")
        raise HTTPException(status_code=500, detail="Failed to get hourly distribution")


@router.get("/{organization_id}/dashboard/summary")
async def get_dashboard_summary(
    organization_id: uuid.UUID,
    org_data: tuple = Depends(require_read_permission()),
    db: Session = Depends(get_db)
):
    """
    Get a quick dashboard summary with key metrics.
    
    Args:
        organization_id: Organization ID
        org_data: Organization and membership data
        db: Database session
        
    Returns:
        Dashboard summary
    """
    try:
        organization, membership = org_data
        
        # Get overview for last 30 days
        overview = await analytics_service.get_dashboard_overview(
            db=db,
            organization_id=str(organization.id),
            date_from=datetime.utcnow() - timedelta(days=30)
        )
        
        # Get trends for last 7 days
        trends = await analytics_service.get_call_trends(
            db=db,
            organization_id=str(organization.id),
            period="day",
            days=7
        )
        
        # Get hourly distribution for last 7 days
        hourly = await analytics_service.get_hourly_distribution(
            db=db,
            organization_id=str(organization.id),
            days=7
        )
        
        return {
            "summary": {
                "total_calls_30d": overview["overview"]["total_calls"],
                "inbound_calls_30d": overview["overview"]["inbound_calls"],
                "outbound_calls_30d": overview["overview"]["outbound_calls"],
                "total_minutes_30d": overview["duration_metrics"]["total_minutes"],
                "total_cost_30d": overview["cost_metrics"]["total_cost_usd"],
                "success_rate_30d": overview["overview"]["success_rate"],
                "avg_quality_30d": overview["quality_metrics"]["average_quality_score"]
            },
            "recent_trends": trends["trends"][-7:] if trends["trends"] else [],  # Last 7 days
            "hourly_pattern": hourly["hourly_distribution"],
            "top_phone_numbers": overview["phone_number_usage"][:5],  # Top 5
            "top_assistants": overview["assistant_usage"][:5]  # Top 5
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard summary")
