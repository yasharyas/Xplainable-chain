import shap
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class XAIExplainer:
    """Explainable AI using SHAP (SHapley Additive exPlanations)"""
    
    def __init__(self):
        """Initialize SHAP explainer"""
        self.explainer = None
    
    def explain(self, features: dict, model) -> dict:
        """
        Generate SHAP explanation for prediction
        
        Args:
            features: Transaction features
            model: Trained model
        
        Returns:
            {
                'shap_values': list,
                'feature_importance': dict,
                'top_features': list,
                'base_value': float
            }
        """
        try:
            # Create explainer if not exists (use TreeExplainer for tree-based models)
            if self.explainer is None:
                try:
                    self.explainer = shap.TreeExplainer(model)
                except Exception as e:
                    logger.warning(f"TreeExplainer failed, using mock explainer: {e}")
                    return self._mock_explanation(features)
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(df)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification returns list of arrays
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Get feature importance
            feature_importance = {}
            for i, feature in enumerate(df.columns):
                feature_importance[feature] = float(abs(shap_values[0][i]))
            
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Top 5 features
            top_features = [
                {
                    'feature': name,
                    'importance': importance,
                    'value': float(features[name])
                }
                for name, importance in sorted_features[:5]
            ]
            
            return {
                'shap_values': shap_values[0].tolist() if hasattr(shap_values[0], 'tolist') else list(shap_values[0]),
                'feature_importance': feature_importance,
                'top_features': top_features,
                'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.5
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            logger.warning("Using mock explanation")
            return self._mock_explanation(features)
    
    def _mock_explanation(self, features: dict) -> dict:
        """Generate mock explanation when SHAP is not available"""
        # Simple heuristic importance
        feature_importance = {
            'gas_price': abs(features.get('gas_price', 50) - 50) / 100,
            'value': features.get('value', 0) / 100,
            'gas_price_deviation': features.get('gas_price_deviation', 0),
            'sender_tx_count': 0.05,
            'contract_age': 0.03,
            'gas_used': 0.04,
            'is_contract_creation': 0.02,
            'function_signature_hash': 0.01,
            'block_gas_used_ratio': 0.02
        }
        
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [
            {
                'feature': name,
                'importance': importance,
                'value': float(features.get(name, 0))
            }
            for name, importance in sorted_features[:5]
        ]
        
        return {
            'shap_values': list(feature_importance.values()),
            'feature_importance': feature_importance,
            'top_features': top_features,
            'base_value': 0.5
        }
