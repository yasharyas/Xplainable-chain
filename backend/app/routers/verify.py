from fastapi import APIRouter, HTTPException, Path
import logging
from app.models.schemas import VerifyResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{tx_hash}", response_model=VerifyResponse)
async def verify_explanation(
    tx_hash: str = Path(..., description="Transaction hash to verify")
):
    """
    Verify if an explanation exists on-chain for a transaction
    
    Args:
        tx_hash: Transaction hash to verify
    
    Returns:
        VerifyResponse with verification status
    """
    try:
        logger.info(f"Verifying explanation for: {tx_hash}")
        
        # TODO: Query blockchain contract to check if explanation exists
        # For now, return mock data
        
        return VerifyResponse(
            tx_hash=tx_hash,
            exists=True,
            verified=False,
            ipfs_hash="QmMockHash123...",
            risk_score=75,
            auditor="0x123...abc",
            timestamp=1609459200
        )
        
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
