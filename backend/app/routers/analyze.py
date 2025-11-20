from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging
from datetime import datetime
import re

from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.models.ai_detector import AIDetector
from app.models.xai_explainer import XAIExplainer
from app.services.blockchain import BlockchainService
from app.services.ipfs import IPFSService
from app.utils.feature_engineering import extract_features

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services (singleton pattern)
ai_detector = AIDetector()
xai_explainer = XAIExplainer()
blockchain_service = BlockchainService()
ipfs_service = IPFSService()

def validate_transaction_hash(tx_hash: str) -> bool:
    """
    Validate Ethereum/Polygon transaction hash format.
    Must be 66 characters long and start with '0x'.
    """
    if not tx_hash:
        return False
    
    # Check if it starts with 0x and is 66 characters (0x + 64 hex chars)
    if not tx_hash.startswith('0x'):
        return False
    
    if len(tx_hash) != 66:
        return False
    
    # Check if remaining characters are valid hexadecimal
    try:
        int(tx_hash[2:], 16)
        return True
    except ValueError:
        return False

@router.post("/", response_model=AnalyzeResponse)
async def analyze_transaction(request: AnalyzeRequest):
    """
    Analyze a blockchain transaction for malicious activity.
    Returns AI prediction + SHAP explanation + on-chain proof.
    
    Args:
        request: AnalyzeRequest with tx_hash and network
    
    Returns:
        AnalyzeResponse with prediction, explanation, and storage hashes
    """
    try:
        logger.info(f"Analyzing transaction: {request.tx_hash}")
        
        # 1. Validate transaction hash format
        if not validate_transaction_hash(request.tx_hash):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid transaction hash format. Must be 66 characters starting with '0x' (e.g., 0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060)"
            )
        
        # 2. REJECT mock data - always fetch from blockchain
        if request.transaction_data:
            logger.warning(f"⚠️  Rejecting provided transaction_data - will fetch from blockchain instead")
            # Don't use provided data - this was the root cause of accepting invalid hashes
        
        # 3. Fetch transaction data from blockchain (REQUIRED)
        tx_data = await blockchain_service.get_transaction(request.tx_hash, request.network)
        if not tx_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Transaction {request.tx_hash} not found on {request.network} network. Please verify the hash and network are correct."
            )
        
        logger.info(f" Fetched real transaction data from {request.network}")
        
        # 4. Extract features for ML model
        features = extract_features(tx_data)
        logger.info(f"Extracted features: {features}")
        
        # 5. Run AI detection
        prediction = ai_detector.predict(features)
        is_malicious = prediction['is_malicious']
        confidence = prediction['confidence']
        risk_score = int(confidence * 100)
        
        logger.info(f"Prediction: malicious={is_malicious}, confidence={confidence}")
        
        # 6. Generate SHAP explanation
        shap_explanation = xai_explainer.explain(features, ai_detector.model)
        
        # 7. Create explanation JSON
        explanation_data = {
            "tx_hash": request.tx_hash,
            "risk_score": risk_score,
            "is_malicious": is_malicious,
            "confidence": confidence,
            "shap_values": shap_explanation['shap_values'],
            "feature_importance": shap_explanation['feature_importance'],
            "top_features": shap_explanation['top_features'],
            "base_value": shap_explanation.get('base_value', 0.5),
            "model_version": "xgboost_v1.0",
            "timestamp": datetime.now().isoformat(),
            "network": request.network
        }
        
        # 8. Upload to IPFS (optional - graceful degradation)
        ipfs_hash = None
        try:
            ipfs_hash = await ipfs_service.upload_json(explanation_data)
            logger.info(f" Uploaded to IPFS: {ipfs_hash}")
        except HTTPException as e:
            if e.status_code == 503:
                logger.warning(f"⚠️  IPFS upload failed (Pinata JWT lacks scopes), continuing without IPFS storage")
                ipfs_hash = "IPFS_UNAVAILABLE_NO_SCOPES"
            else:
                raise
        except Exception as e:
            logger.warning(f"⚠️  IPFS upload failed: {str(e)}, continuing without IPFS storage")
            ipfs_hash = "IPFS_UNAVAILABLE"
        
        # 9. Store on blockchain (optional - graceful degradation)
        blockchain_hash = None
        try:
            if ipfs_hash and not ipfs_hash.startswith("IPFS_UNAVAILABLE"):
                blockchain_hash = await blockchain_service.store_explanation(
                    tx_hash=request.tx_hash,
                    ipfs_hash=ipfs_hash,
                    risk_score=risk_score
                )
                logger.info(f" Stored on blockchain: {blockchain_hash}")
            else:
                logger.warning("⚠️  Skipping blockchain storage (no valid IPFS hash)")
                blockchain_hash = "BLOCKCHAIN_UNAVAILABLE_NO_IPFS"
        except Exception as e:
            logger.warning(f"⚠️  Blockchain storage failed: {str(e)}, continuing without blockchain storage")
            blockchain_hash = "BLOCKCHAIN_UNAVAILABLE"
        
        return AnalyzeResponse(
            tx_hash=request.tx_hash,
            is_malicious=is_malicious,
            risk_score=risk_score,
            confidence=confidence,
            explanation=explanation_data,
            ipfs_hash=ipfs_hash,
            blockchain_hash=blockchain_hash,
            features=features,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
