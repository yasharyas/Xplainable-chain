from fastapi import APIRouter, Query
from typing import List
import logging
from app.models.schemas import AuditLogEntry

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[AuditLogEntry])
async def get_audit_trail(
    limit: int = Query(default=10, ge=1, le=100, description="Number of entries to return"),
    skip: int = Query(default=0, ge=0, description="Number of entries to skip")
):
    """
    Get audit trail of analyzed transactions
    
    Args:
        limit: Maximum number of entries to return
        skip: Number of entries to skip (pagination)
    
    Returns:
        List of audit log entries
    """
    try:
        logger.info(f"Fetching audit trail: limit={limit}, skip={skip}")
        
        # TODO: Query MongoDB for audit logs
        # For now, return mock data
        
        mock_entries = [
            AuditLogEntry(
                tx_hash="0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060",
                risk_score=85,
                is_malicious=True,
                analyzed_at="2024-01-15T10:30:00Z",
                blockchain_hash="0xabc123...",
                ipfs_hash="QmTest123..."
            ),
            AuditLogEntry(
                tx_hash="0x9fc76417374aa880d4449a1f7f31ec597f00b1f6f3dd2d66f4c9c6c445836d8b",
                risk_score=25,
                is_malicious=False,
                analyzed_at="2024-01-15T09:15:00Z",
                blockchain_hash="0xdef456...",
                ipfs_hash="QmTest456..."
            )
        ]
        
        return mock_entries[skip:skip+limit]
        
    except Exception as e:
        logger.error(f"Audit trail error: {str(e)}")
        return []
