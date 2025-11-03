from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging
from datetime import datetime

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
        
        # 1. Get transaction data (either from request or fetch from blockchain)
        if request.transaction_data:
            # Use provided transaction data
            tx_data = request.transaction_data
            logger.info("Using provided transaction data")
        else:
            # Fetch transaction data from blockchain
            tx_data = await blockchain_service.get_transaction(request.tx_hash, request.network)
            if not tx_data:
                raise HTTPException(status_code=404, detail="Transaction not found")
        
        # 2. Extract features for ML model
        features = extract_features(tx_data)
        logger.info(f"Extracted features: {features}")
        
        # 3. Run AI detection
        prediction = ai_detector.predict(features)
        is_malicious = prediction['is_malicious']
        confidence = prediction['confidence']
        risk_score = int(confidence * 100)
        
        logger.info(f"Prediction: malicious={is_malicious}, confidence={confidence}")
        
        # 4. Generate SHAP explanation
        shap_explanation = xai_explainer.explain(features, ai_detector.model)
        
        # 5. Create explanation JSON
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
        
        # 6. Upload to IPFS
        ipfs_hash = await ipfs_service.upload_json(explanation_data)
        logger.info(f"Uploaded to IPFS: {ipfs_hash}")
        
        # 7. Store on blockchain
        blockchain_hash = await blockchain_service.store_explanation(
            tx_hash=request.tx_hash,
            ipfs_hash=ipfs_hash,
            risk_score=risk_score
        )
        logger.info(f"Stored on blockchain: {blockchain_hash}")
        
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
