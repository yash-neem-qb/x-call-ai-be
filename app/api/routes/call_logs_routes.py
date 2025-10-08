"""
Call logs API routes.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud import create_call_log, get_call_logs, get_call_log, update_call_log, delete_call_log
from app.models.call_logs_schemas import CallLogCreate, CallLogUpdate, CallLogResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/call-logs", tags=["call-logs"])


@router.post("/", response_model=CallLogResponse)
async def create_call_log_entry(
    call_data: CallLogCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new call log entry.
    """
    try:
        call = create_call_log(db, call_data.dict())
        return CallLogResponse.from_orm(call)
    except Exception as e:
        logger.error(f"Error creating call log: {e}")
        raise HTTPException(status_code=500, detail="Failed to create call log")


@router.get("/", response_model=List[CallLogResponse])
async def get_call_logs_list(
    organization_id: uuid.UUID = Query(..., description="Organization ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
):
    """
    Get call logs for an organization with pagination.
    """
    try:
        calls = get_call_logs(db, organization_id, skip, limit)
        return [CallLogResponse.from_orm(call) for call in calls]
    except Exception as e:
        logger.error(f"Error fetching call logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch call logs")


@router.get("/{call_id}", response_model=CallLogResponse)
async def get_call_log_entry(
    call_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific call log by ID.
    """
    call = get_call_log(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call log not found")
    return CallLogResponse.from_orm(call)


@router.put("/{call_id}", response_model=CallLogResponse)
async def update_call_log_entry(
    call_id: uuid.UUID,
    call_data: CallLogUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a call log entry.
    """
    call = update_call_log(db, call_id, call_data.dict(exclude_unset=True))
    if not call:
        raise HTTPException(status_code=404, detail="Call log not found")
    return CallLogResponse.from_orm(call)


@router.delete("/{call_id}")
async def delete_call_log_entry(
    call_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a call log entry.
    """
    success = delete_call_log(db, call_id)
    if not success:
        raise HTTPException(status_code=404, detail="Call log not found")
    return {"message": "Call log deleted successfully"}


@router.delete("/bulk")
async def delete_multiple_call_logs(
    call_ids: List[uuid.UUID],
    db: Session = Depends(get_db)
):
    """
    Delete multiple call log entries.
    """
    try:
        deleted_count = 0
        failed_ids = []
        
        for call_id in call_ids:
            success = delete_call_log(db, call_id)
            if success:
                deleted_count += 1
            else:
                failed_ids.append(str(call_id))
        
        if failed_ids:
            return {
                "message": f"Deleted {deleted_count} call logs, {len(failed_ids)} failed",
                "deleted_count": deleted_count,
                "failed_ids": failed_ids
            }
        
        return {
            "message": f"Successfully deleted {deleted_count} call logs",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error deleting multiple call logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete call logs")


@router.post("/{call_id}/end")
async def end_call_log(
    call_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    End a call and update its status.
    """
    call = get_call_log(db, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call log not found")
    
    # Calculate duration if call was started
    duration_seconds = None
    if call.started_at:
        duration_seconds = int((datetime.utcnow() - call.started_at).total_seconds())
    
    update_data = {
        "status": "completed",
        "endedAt": datetime.utcnow(),
        "durationSeconds": duration_seconds
    }
    
    updated_call = update_call_log(db, call_id, update_data)
    return CallLogResponse.from_orm(updated_call)
