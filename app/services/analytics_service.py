"""
Analytics service for dashboard metrics and reporting.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, asc
from sqlalchemy.sql import text

from app.db.models import Call, PhoneNumber, Assistant, Organization, User, CallStatus, CallDirection

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for generating analytics and dashboard metrics."""
    
    def __init__(self):
        pass
    
    async def get_dashboard_overview(
        self, 
        db: Session, 
        organization_id: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        assistant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive dashboard overview metrics.
        
        Args:
            db: Database session
            organization_id: Organization ID
            date_from: Start date filter
            date_to: End date filter
            assistant_id: Optional assistant ID filter
            
        Returns:
            Dictionary with dashboard metrics
        """
        try:
            # Build base query
            base_query = db.query(Call).filter(Call.organization_id == organization_id)
            
            if date_from:
                base_query = base_query.filter(Call.created_at >= date_from)
            if date_to:
                base_query = base_query.filter(Call.created_at <= date_to)
            if assistant_id:
                base_query = base_query.filter(Call.assistant_id == assistant_id)
            
            # Total calls
            total_calls = base_query.count()
            
            # Calls by direction
            inbound_calls = base_query.filter(Call.direction == CallDirection.INBOUND).count()
            outbound_calls = base_query.filter(Call.direction == CallDirection.OUTBOUND).count()
            
            # Calls by status
            completed_calls = base_query.filter(Call.status == CallStatus.COMPLETED).count()
            failed_calls = base_query.filter(Call.status == CallStatus.FAILED).count()
            busy_calls = base_query.filter(Call.status == CallStatus.BUSY).count()
            no_answer_calls = base_query.filter(Call.status == CallStatus.NO_ANSWER).count()
            
            # Duration metrics
            duration_stats = base_query.filter(
                and_(
                    Call.duration_seconds.isnot(None),
                    Call.duration_seconds > 0
                )
            ).with_entities(
                func.avg(Call.duration_seconds).label('avg_duration'),
                func.sum(Call.duration_seconds).label('total_duration'),
                func.min(Call.duration_seconds).label('min_duration'),
                func.max(Call.duration_seconds).label('max_duration')
            ).first()
            
            # Cost metrics
            cost_stats = base_query.filter(
                and_(
                    Call.cost_usd.isnot(None),
                    Call.cost_usd > 0
                )
            ).with_entities(
                func.sum(Call.cost_usd).label('total_cost'),
                func.avg(Call.cost_usd).label('avg_cost'),
                func.min(Call.cost_usd).label('min_cost'),
                func.max(Call.cost_usd).label('max_cost')
            ).first()
            
            # Quality metrics
            quality_stats = base_query.filter(
                and_(
                    Call.quality_score.isnot(None),
                    Call.quality_score > 0
                )
            ).with_entities(
                func.avg(Call.quality_score).label('avg_quality'),
                func.avg(Call.sentiment_score).label('avg_sentiment'),
                func.avg(Call.satisfaction_rating).label('avg_satisfaction')
            ).first()
            
            # Phone number usage
            phone_usage = db.query(
                PhoneNumber.id,
                PhoneNumber.phone_number,
                PhoneNumber.friendly_name,
                PhoneNumber.assistant_id,
                func.count(Call.id).label('call_count'),
                func.sum(Call.duration_seconds).label('total_duration'),
                func.sum(Call.cost_usd).label('total_cost')
            ).outerjoin(Call, PhoneNumber.id == Call.phone_number_id)\
            .filter(PhoneNumber.organization_id == organization_id)\
            .group_by(
                PhoneNumber.id,
                PhoneNumber.phone_number,
                PhoneNumber.friendly_name,
                PhoneNumber.assistant_id
            )\
            .order_by(desc('call_count')).limit(10).all()
            
            # Assistant usage
            assistant_usage = db.query(
                Assistant.name,
                func.count(Call.id).label('call_count'),
                func.sum(Call.duration_seconds).label('total_duration'),
                func.avg(Call.quality_score).label('avg_quality')
            ).join(Call, Assistant.id == Call.assistant_id)\
            .filter(Call.organization_id == organization_id)\
            .group_by(Assistant.id, Assistant.name)\
            .order_by(desc('call_count')).limit(10).all()
            
            return {
                "overview": {
                    "total_calls": total_calls,
                    "inbound_calls": inbound_calls,
                    "outbound_calls": outbound_calls,
                    "completed_calls": completed_calls,
                    "failed_calls": failed_calls,
                    "busy_calls": busy_calls,
                    "no_answer_calls": no_answer_calls,
                    "success_rate": float(completed_calls / total_calls * 100) if total_calls > 0 else 0.0
                },
                "duration_metrics": {
                    "total_minutes": float(duration_stats.total_duration) / 60 if duration_stats.total_duration else 0.0,
                    "average_duration_seconds": float(duration_stats.avg_duration) if duration_stats.avg_duration else 0,
                    "average_duration_minutes": float(duration_stats.avg_duration) / 60 if duration_stats.avg_duration else 0.0,
                    "min_duration_seconds": duration_stats.min_duration or 0,
                    "max_duration_seconds": duration_stats.max_duration or 0
                },
                "cost_metrics": {
                    "total_cost_usd": float(cost_stats.total_cost) if cost_stats.total_cost else 0,
                    "average_cost_usd": float(cost_stats.avg_cost) if cost_stats.avg_cost else 0,
                    "min_cost_usd": float(cost_stats.min_cost) if cost_stats.min_cost else 0,
                    "max_cost_usd": float(cost_stats.max_cost) if cost_stats.max_cost else 0,
                    "cost_per_minute": float(cost_stats.total_cost) / (float(duration_stats.total_duration) / 60) if (cost_stats.total_cost and duration_stats.total_duration) else 0.0
                },
                "quality_metrics": {
                    "average_quality_score": float(quality_stats.avg_quality) if quality_stats.avg_quality else 0,
                    "average_sentiment_score": float(quality_stats.avg_sentiment) if quality_stats.avg_sentiment else 0,
                    "average_satisfaction_rating": float(quality_stats.avg_satisfaction) if quality_stats.avg_satisfaction else 0
                },
                "phone_number_usage": [
                    {
                        "phone_number_id": str(usage.id),
                        "phone_number": usage.phone_number,
                        "friendly_name": usage.friendly_name,
                        "assistant_id": str(usage.assistant_id) if usage.assistant_id else None,
                        "call_count": usage.call_count,
                        "total_duration_minutes": float(usage.total_duration) / 60 if usage.total_duration else 0.0,
                        "total_cost_usd": float(usage.total_cost) if usage.total_cost else 0
                    }
                    for usage in phone_usage
                ],
                "assistant_usage": [
                    {
                        "assistant_name": usage.name,
                        "call_count": usage.call_count,
                        "total_duration_minutes": float(usage.total_duration) / 60 if usage.total_duration else 0.0,
                        "average_quality_score": float(usage.avg_quality) if usage.avg_quality else 0
                    }
                    for usage in assistant_usage
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            raise
    
    async def get_call_trends(
        self,
        db: Session,
        organization_id: str,
        period: str = "day",  # day, week, month
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get call trends over time.
        
        Args:
            db: Database session
            organization_id: Organization ID
            period: Aggregation period (day, week, month)
            days: Number of days to look back
            
        Returns:
            Dictionary with trend data
        """
        try:
            date_from = datetime.utcnow() - timedelta(days=days)
            
            # Build date truncation based on period
            if period == "day":
                date_trunc = func.date_trunc('day', Call.created_at)
            elif period == "week":
                date_trunc = func.date_trunc('week', Call.created_at)
            elif period == "month":
                date_trunc = func.date_trunc('month', Call.created_at)
            else:
                date_trunc = func.date_trunc('day', Call.created_at)
            
            # Get call trends
            trends = db.query(
                date_trunc.label('date'),
                func.count(Call.id).label('total_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.INBOUND).label('inbound_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.OUTBOUND).label('outbound_calls'),
                func.count(Call.id).filter(Call.status == CallStatus.COMPLETED).label('completed_calls'),
                func.sum(Call.duration_seconds).label('total_duration'),
                func.sum(Call.cost_usd).label('total_cost'),
                func.avg(Call.quality_score).label('avg_quality')
            ).filter(
                and_(
                    Call.organization_id == organization_id,
                    Call.created_at >= date_from
                )
            ).group_by(date_trunc).order_by('date').all()
            
            return {
                "period": period,
                "days": days,
                "trends": [
                    {
                        "date": trend.date.isoformat(),
                        "total_calls": trend.total_calls,
                        "inbound_calls": trend.inbound_calls,
                        "outbound_calls": trend.outbound_calls,
                        "completed_calls": trend.completed_calls,
                        "total_duration_minutes": float(trend.total_duration) / 60 if trend.total_duration else 0.0,
                        "total_cost_usd": float(trend.total_cost) if trend.total_cost else 0,
                        "average_quality_score": float(trend.avg_quality) if trend.avg_quality else 0
                    }
                    for trend in trends
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting call trends: {e}")
            raise
    
    async def get_phone_number_analytics(
        self,
        db: Session,
        organization_id: str,
        phone_number_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed phone number analytics.
        
        Args:
            db: Database session
            organization_id: Organization ID
            phone_number_id: Optional specific phone number ID
            
        Returns:
            Dictionary with phone number analytics
        """
        try:
            base_query = db.query(Call).filter(Call.organization_id == organization_id)
            
            if phone_number_id:
                base_query = base_query.filter(Call.phone_number_id == phone_number_id)
            
            # Get phone numbers with their stats
            phone_stats = db.query(
                PhoneNumber.id,
                PhoneNumber.phone_number,
                PhoneNumber.friendly_name,
                PhoneNumber.assistant_id,
                func.count(Call.id).label('total_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.INBOUND).label('inbound_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.OUTBOUND).label('outbound_calls'),
                func.count(Call.id).filter(Call.status == CallStatus.COMPLETED).label('completed_calls'),
                func.sum(Call.duration_seconds).label('total_duration'),
                func.sum(Call.cost_usd).label('total_cost'),
                func.avg(Call.quality_score).label('avg_quality')
            ).outerjoin(Call, PhoneNumber.id == Call.phone_number_id)\
            .filter(PhoneNumber.organization_id == organization_id)\
            .group_by(
                PhoneNumber.id,
                PhoneNumber.phone_number,
                PhoneNumber.friendly_name,
                PhoneNumber.assistant_id
            )\
            .all()
            
            return {
                "phone_numbers": [
                    {
                        "id": str(stat.id),
                        "phone_number": stat.phone_number,
                        "friendly_name": stat.friendly_name,
                        "assistant_id": str(stat.assistant_id) if stat.assistant_id else None,
                        "total_calls": stat.total_calls,
                        "inbound_calls": stat.inbound_calls,
                        "outbound_calls": stat.outbound_calls,
                        "completed_calls": stat.completed_calls,
                        "success_rate": float(stat.completed_calls / stat.total_calls * 100) if stat.total_calls > 0 else 0.0,
                        "total_duration_minutes": float(stat.total_duration) / 60 if stat.total_duration else 0.0,
                        "total_cost_usd": float(stat.total_cost) if stat.total_cost else 0,
                        "average_quality_score": float(stat.avg_quality) if stat.avg_quality else 0
                    }
                    for stat in phone_stats
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting phone number analytics: {e}")
            raise
    
    async def get_assistant_analytics(
        self,
        db: Session,
        organization_id: str,
        assistant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed assistant analytics.
        
        Args:
            db: Database session
            organization_id: Organization ID
            assistant_id: Optional specific assistant ID
            
        Returns:
            Dictionary with assistant analytics
        """
        try:
            base_query = db.query(Call).filter(Call.organization_id == organization_id)
            
            if assistant_id:
                base_query = base_query.filter(Call.assistant_id == assistant_id)
            
            # Get assistants with their stats
            assistant_stats = db.query(
                Assistant.id,
                Assistant.name,
                func.count(Call.id).label('total_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.INBOUND).label('inbound_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.OUTBOUND).label('outbound_calls'),
                func.count(Call.id).filter(Call.status == CallStatus.COMPLETED).label('completed_calls'),
                func.sum(Call.duration_seconds).label('total_duration'),
                func.avg(Call.quality_score).label('avg_quality'),
                func.avg(Call.sentiment_score).label('avg_sentiment'),
                func.avg(Call.satisfaction_rating).label('avg_satisfaction')
            ).join(Call, Assistant.id == Call.assistant_id)\
            .filter(Assistant.organization_id == organization_id)\
            .group_by(Assistant.id, Assistant.name)\
            .all()
            
            return {
                "assistants": [
                    {
                        "id": str(stat.id),
                        "name": stat.name,
                        "total_calls": stat.total_calls,
                        "inbound_calls": stat.inbound_calls,
                        "outbound_calls": stat.outbound_calls,
                        "completed_calls": stat.completed_calls,
                        "success_rate": float(stat.completed_calls / stat.total_calls * 100) if stat.total_calls > 0 else 0.0,
                        "total_duration_minutes": float(stat.total_duration) / 60 if stat.total_duration else 0.0,
                        "average_quality_score": float(stat.avg_quality) if stat.avg_quality else 0,
                        "average_sentiment_score": float(stat.avg_sentiment) if stat.avg_sentiment else 0,
                        "average_satisfaction_rating": float(stat.avg_satisfaction) if stat.avg_satisfaction else 0
                    }
                    for stat in assistant_stats
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting assistant analytics: {e}")
            raise
    
    async def get_hourly_distribution(
        self,
        db: Session,
        organization_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get call distribution by hour of day.
        
        Args:
            db: Database session
            organization_id: Organization ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with hourly distribution
        """
        try:
            date_from = datetime.utcnow() - timedelta(days=days)
            
            hourly_stats = db.query(
                func.extract('hour', Call.created_at).label('hour'),
                func.count(Call.id).label('call_count'),
                func.count(Call.id).filter(Call.direction == CallDirection.INBOUND).label('inbound_calls'),
                func.count(Call.id).filter(Call.direction == CallDirection.OUTBOUND).label('outbound_calls')
            ).filter(
                and_(
                    Call.organization_id == organization_id,
                    Call.created_at >= date_from
                )
            ).group_by(func.extract('hour', Call.created_at)).order_by('hour').all()
            
            # Create array for all 24 hours
            hourly_data = []
            for hour in range(24):
                hour_stat = next((stat for stat in hourly_stats if stat.hour == hour), None)
                hourly_data.append({
                    "hour": int(hour),
                    "call_count": hour_stat.call_count if hour_stat else 0,
                    "inbound_calls": hour_stat.inbound_calls if hour_stat else 0,
                    "outbound_calls": hour_stat.outbound_calls if hour_stat else 0
                })
            
            return {
                "period_days": days,
                "hourly_distribution": hourly_data
            }
            
        except Exception as e:
            logger.error(f"Error getting hourly distribution: {e}")
            raise


# Global instance
analytics_service = AnalyticsService()
