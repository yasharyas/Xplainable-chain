"""
Causal Analysis API Router
Provides endpoints for causal XAI explanations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze/causal", tags=["causal-analysis"])


class CausalAnalysisRequest(BaseModel):
    """Request model for causal analysis"""
    transaction_hash: Optional[str] = None
    features: Dict[str, float]
    treatment_features: Optional[List[str]] = None


class CausalAnalysisResponse(BaseModel):
    """Response model for causal analysis"""
    causal_effects: Dict
    correlations: Dict
    comparison: List[Dict]
    confounders: Dict
    interpretation: str
    causal_graph: Optional[Dict] = None


@router.post("/", response_model=CausalAnalysisResponse)
async def analyze_causal_effects(request: CausalAnalysisRequest):
    """
    Perform causal analysis on transaction features
    
    Returns:
    - Causal effects (how features CAUSE fraud)
    - Correlations (how features CORRELATE with fraud)
    - Comparison showing spurious vs genuine relationships
    - Confounding variables that create spurious correlations
    - Human-readable interpretation
    """
    try:
        from app.models.causal_xai_explainer import CausalXAIExplainer
        from app.models.ai_detector import AIDetector
        from pymongo import MongoClient
        import pandas as pd
        import os
        
        logger.info(f"Causal analysis request for features: {list(request.features.keys())}")
        
        # Get historical data from MongoDB
        try:
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
            client = MongoClient(mongo_uri)
            db = client["xai_chain"]
            collection = db["fraud_predictions"]
            
            # Fetch recent predictions to build training dataset
            recent_predictions = list(collection.find({}, {
                '_id': 0,
                'transaction_data': 1,
                'prediction': 1,
                'prediction_score': 1
            }).limit(500))
            
            logger.info(f"Retrieved {len(recent_predictions)} historical transactions from MongoDB")
            
            # Convert to DataFrame for causal analysis
            if len(recent_predictions) > 100:
                training_data = pd.DataFrame([
                    {
                        **pred['transaction_data'].get('features', {}),
                        'malicious': 1 if pred['prediction'] == 'Malicious' else 0,
                        'fraud_score': pred.get('prediction_score', 0)
                    }
                    for pred in recent_predictions
                    if 'transaction_data' in pred and 'features' in pred['transaction_data']
                ])
                
                # Add derived features
                if 'gas_price' in training_data.columns:
                    median_gas = training_data['gas_price'].median()
                    training_data['gas_price_deviation'] = abs(training_data['gas_price'] - median_gas) / median_gas
                
                logger.info(f"Prepared training dataset with {len(training_data)} samples and {len(training_data.columns)} features")
            else:
                logger.warning("Insufficient historical data, will use synthetic data")
                training_data = None
                
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Using synthetic data.")
            training_data = None
        
        # Initialize explainer
        explainer = CausalXAIExplainer()
        
        # Perform causal analysis with historical or synthetic data
        analysis = explainer.explain_causal_effects(
            features=request.features,
            training_data=training_data,
            treatment_features=request.treatment_features,
            use_ml_model=True
        )
        
        data_type = "historical" if training_data is not None and len(training_data) > 100 else "synthetic"
        logger.info(f"Causal analysis completed using {data_type} data: {len(analysis.get('causal_effects', {}))} effects estimated")
        
        return CausalAnalysisResponse(
            causal_effects=analysis['causal_effects'],
            correlations=analysis['correlations'],
            comparison=analysis['comparison'],
            confounders=analysis['confounders'],
            interpretation=analysis['interpretation'],
            causal_graph=None  # Will add visualization data later
        )
    
    except ImportError as e:
        logger.error(f"Causal analysis dependencies not installed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Causal analysis feature requires additional dependencies. Please rebuild the Docker container."
        )
    except Exception as e:
        logger.error(f"Causal analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Causal analysis failed: {str(e)}"
        )


@router.get("/graph")
async def get_causal_graph():
    """
    Get the causal graph structure for visualization
    
    Returns:
    - Nodes (features, confounders, outcome)
    - Edges (causal relationships)
    - Graph statistics
    """
    try:
        from app.models.causal_xai_explainer import CausalXAIExplainer
        
        explainer = CausalXAIExplainer()
        graph_structure = explainer.get_causal_graph_structure()
        
        return {
            "graph": graph_structure,
            "description": "Causal graph showing relationships between blockchain features and fraud"
        }
    
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Causal analysis feature requires additional dependencies"
        )
    except Exception as e:
        logger.error(f"Error fetching causal graph: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve causal graph: {str(e)}"
        )


@router.post("/compare")
async def compare_shap_vs_causal(request: CausalAnalysisRequest):
    """
    Compare SHAP (correlation-based) vs Causal explanations
    
    Shows the difference between:
    - SHAP: Feature importance based on correlation
    - Causal: True causal effects after controlling for confounders
    """
    try:
        from app.models.causal_xai_explainer import CausalXAIExplainer
        from app.models.xai_explainer import XAIExplainer
        
        # Get SHAP explanations
        shap_explainer = XAIExplainer()
        # Note: Would need model reference here - simplified for now
        
        # Get Causal explanations
        causal_explainer = CausalXAIExplainer()
        causal_analysis = causal_explainer.explain_causal_effects(
            features=request.features,
            treatment_features=request.treatment_features
        )
        
        # Build comparison
        comparison_data = {
            "message": "SHAP shows correlation, Causal shows causation",
            "key_differences": causal_analysis['comparison'],
            "recommendation": "Use causal effects for decision-making, SHAP for quick insights"
        }
        
        return comparison_data
    
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Comparison feature requires causal analysis dependencies"
        )
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@router.get("/info")
async def get_causal_analysis_info():
    """
    Get information about causal analysis capabilities
    """
    return {
        "feature": "Causal Explainable AI (Causal XAI)",
        "purpose": "Distinguish correlation from causation in fraud detection",
        "methods": [
            "Backdoor Adjustment (controlling for confounders)",
            "Instrumental Variables",
            "Regression Discontinuity"
        ],
        "advantages": [
            "Identifies genuine causal relationships",
            "Reveals spurious correlations",
            "Provides actionable insights",
            "Robust to confounding"
        ],
        "endpoints": {
            "POST /api/analyze/causal/": "Perform full causal analysis",
            "GET /api/analyze/causal/graph": "View causal graph structure",
            "POST /api/analyze/causal/compare": "Compare SHAP vs Causal"
        },
        "research_contribution": "Novel application of causal inference to blockchain fraud detection"
    }
