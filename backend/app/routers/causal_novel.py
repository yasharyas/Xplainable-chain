"""
NOVEL API Endpoints: Counterfactual Explanations & Intervention Recommendations
Research contribution: Actionable fraud prevention based on causal mechanisms
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from app.utils.mongodb_fetcher import fetch_training_data_from_mongodb

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze/causal/novel", tags=["novel-causal-xai"])


class CounterfactualRequest(BaseModel):
    """Request for counterfactual explanation"""
    features: Dict[str, float]
    current_fraud_prob: float
    target_fraud_prob: Optional[float] = 0.1
    max_interventions: Optional[int] = 3


class CounterfactualResponse(BaseModel):
    """Response with counterfactual scenarios"""
    current_features: Dict
    counterfactuals: List[Dict]
    recommendations: str


class InterventionRequest(BaseModel):
    """Request for intervention recommendations"""
    features: Dict[str, float]
    current_fraud_prob: float


class InterventionResponse(BaseModel):
    """Response with actionable interventions"""
    current_risk: float
    interventions: List[Dict]
    top_recommendation: Dict


class CATERequest(BaseModel):
    """Request for heterogeneous treatment effect analysis"""
    treatment: str = "gas_price"
    outcome: str = "malicious"


class CATEResponse(BaseModel):
    """Response with conditional treatment effects"""
    treatment: str
    outcome: str
    heterogeneity_analysis: Dict
    interpretation: str


@router.post("/counterfactuals", response_model=CounterfactualResponse)
async def generate_counterfactuals(request: CounterfactualRequest):
    """
    NOVEL: Generate counterfactual explanations
    
    Answers: "What if gas_price had been different? Would this transaction still be fraud?"
    
    Research contribution: Provides actionable insights for fraud prevention
    """
    try:
        from app.models.causal_xai_explainer import CausalXAIExplainer
        
        # ROOT FIX: Fetch real training data from MongoDB
        training_data = fetch_training_data_from_mongodb(limit=500)
        logger.info(f"Using {'real MongoDB' if training_data is not None else 'synthetic'} data for counterfactuals")
        
        explainer = CausalXAIExplainer(use_data_driven_discovery=True)
        
        # Generate counterfactuals with real training data
        counterfactuals = explainer.generate_counterfactuals(
            features=request.features,
            target_outcome=request.target_fraud_prob,
            max_interventions=request.max_interventions,
            training_data=training_data  # Pass real MongoDB data
        )
        
        if not counterfactuals:
            return CounterfactualResponse(
                current_features=request.features,
                counterfactuals=[],
                recommendations="Counterfactual analysis unavailable - insufficient causal structure"
            )
        
        # Generate human-readable recommendations
        recommendations_text = " Counterfactual Analysis Results:\n\n"
        
        for i, cf in enumerate(counterfactuals, 1):
            intervention = cf['intervention']
            effect = cf['causal_effect']
            
            recommendations_text += f"{i}. {cf['recommendation']}\n"
            recommendations_text += f"   Intervention: {intervention}\n"
            recommendations_text += f"   Expected fraud reduction: {abs(effect)*100:.1f}%\n\n"
        
        return CounterfactualResponse(
            current_features=request.features,
            counterfactuals=counterfactuals,
            recommendations=recommendations_text
        )
        
    except Exception as e:
        logger.error(f"Counterfactual generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interventions", response_model=InterventionResponse)
async def recommend_interventions(request: InterventionRequest):
    """
    NOVEL: Generate intervention recommendations based on causal mechanisms
    
    Answers: "What should I change to prevent this fraud?"
    
    Research contribution: Goes beyond prediction to provide actionable prevention strategies
    """
    try:
        from app.models.causal_xai_explainer import CausalXAIExplainer
        
        # ROOT FIX: Fetch real training data from MongoDB
        training_data = fetch_training_data_from_mongodb(limit=500)
        logger.info(f"Using {'real MongoDB' if training_data is not None else 'synthetic'} data for interventions")
        
        explainer = CausalXAIExplainer(use_data_driven_discovery=True)
        
        # Get intervention recommendations with real training data
        interventions = explainer.recommend_interventions(
            features=request.features,
            current_fraud_prob=request.current_fraud_prob,
            training_data=training_data  # Pass real MongoDB data
        )
        
        if not interventions:
            raise HTTPException(
                status_code=503,
                detail="Intervention recommendations unavailable"
            )
        
        # Top recommendation
        top = interventions[0] if interventions else {}
        
        return InterventionResponse(
            current_risk=request.current_fraud_prob,
            interventions=interventions,
            top_recommendation=top
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intervention recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/heterogeneity", response_model=CATEResponse)
async def analyze_treatment_heterogeneity(request: CATERequest):
    """
    NOVEL: Analyze how causal effects vary by transaction type
    
    Uses Conditional Average Treatment Effect (CATE)
    
    Research contribution: Shows that causal effects are heterogeneous - 
    gas_price may matter more for contract creation vs regular transfers
    """
    try:
        from app.models.causal_xai_explainer import CausalXAIExplainer
        
        # ROOT FIX: Fetch real training data from MongoDB
        training_data = fetch_training_data_from_mongodb(limit=1000)
        logger.info(f"Using {'real MongoDB' if training_data is not None else 'synthetic'} data for CATE")
        
        if training_data is None or len(training_data) < 100:
            raise HTTPException(
                status_code=503,
                detail="Insufficient data for heterogeneity analysis. Need at least 100 transactions in MongoDB."
            )
        
        # Run CATE analysis with real MongoDB data
        explainer = CausalXAIExplainer(use_data_driven_discovery=True)
        heterogeneity = explainer.analyze_treatment_heterogeneity(
            data=training_data,  # Use real MongoDB data
            treatment=request.treatment,
            outcome=request.outcome
        )
        
        if not heterogeneity:
            raise HTTPException(
                status_code=503,
                detail="Heterogeneity analysis failed"
            )
        
        # Generate interpretation
        interpretation = f" Heterogeneous Treatment Effects:\n\n"
        interpretation += f"The causal effect of {request.treatment} on {request.outcome} varies significantly by transaction type:\n\n"
        
        for group, effects in heterogeneity.items():
            interpretation += f"**{group}**:\n"
            for subgroup, effect in effects.items():
                interpretation += f"  - {subgroup}: {effect:.4f}\n"
            interpretation += "\n"
        
        interpretation += "This suggests that causal relationships are NOT uniform - they depend on transaction characteristics."
        
        return CATEResponse(
            treatment=request.treatment,
            outcome=request.outcome,
            heterogeneity_analysis=heterogeneity,
            interpretation=interpretation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heterogeneity analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/discovery/info")
async def get_causal_discovery_info():
    """
    Information about novel causal discovery features
    """
    return {
        "title": "Novel Causal XAI Features",
        "research_contributions": [
            {
                "name": "Data-Driven Causal Discovery",
                "description": "Automatically learns causal structure from data using NOTEARS algorithm",
                "novelty": "Replaces manual domain knowledge with data-driven approach",
                "endpoint": "/api/analyze/causal/novel/discover"
            },
            {
                "name": "Counterfactual Explanations",
                "description": "Generates 'what-if' scenarios showing how changes prevent fraud",
                "novelty": "Provides actionable explanations, not just predictions",
                "endpoint": "/api/analyze/causal/novel/counterfactuals"
            },
            {
                "name": "Intervention Recommendations",
                "description": "Suggests specific changes to reduce fraud risk based on causal mechanisms",
                "novelty": "Goes from explanation to prevention",
                "endpoint": "/api/analyze/causal/novel/interventions"
            },
            {
                "name": "Heterogeneous Treatment Effects (CATE)",
                "description": "Shows how causal effects vary by transaction type",
                "novelty": "Reveals that causal relationships are context-dependent",
                "endpoint": "/api/analyze/causal/novel/heterogeneity"
            }
        ],
        "key_differences_from_prior_work": {
            "vs_SHAP": "SHAP shows feature importance (correlation). We show causal effects (causation).",
            "vs_standard_XAI": "Standard XAI explains predictions. We explain mechanisms and suggest interventions.",
            "vs_existing_blockchain_fraud": "Prior work focuses on detection accuracy. We focus on actionable prevention."
        }
    }
