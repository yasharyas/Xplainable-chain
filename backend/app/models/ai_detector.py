import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class AIDetector:
    """AI-based malicious transaction detector using XGBoost"""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize AI detector with pre-trained model
        
        Args:
            model_path: Path to the trained model pickle file
            scaler_path: Path to the feature scaler pickle file
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.load_model()
    
    def load_model(self):
        """Load pre-trained XGBoost model and scaler"""
        try:
            # Default paths
            if self.model_path is None:
                self.model_path = Path(__file__).parent.parent / "ml" / "model.pkl"
            if self.scaler_path is None:
                self.scaler_path = Path(__file__).parent.parent / "ml" / "scaler.pkl"
            
            # Check if files exist
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}")
                logger.warning("Using mock model for testing. Please train model first!")
                self._create_mock_model()
                return
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.feature_names = [
                'gas_price', 'gas_used', 'value', 'gas_price_deviation',
                'sender_tx_count', 'contract_age', 'is_contract_creation',
                'function_signature_hash', 'block_gas_used_ratio'
            ]
            
            logger.info("AI model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using mock model for testing")
            self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock model for testing when real model is not available"""
        class MockModel:
            def predict(self, X):
                # Simple heuristic: high gas price or high value = malicious
                return np.array([1 if (x[0] > 80 or x[2] > 50) else 0 for x in X])
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                return np.array([[1-p, p] for p in predictions])
        
        class MockScaler:
            def transform(self, X):
                return X
        
        self.model = MockModel()
        self.scaler = MockScaler()
        self.feature_names = [
            'gas_price', 'gas_used', 'value', 'gas_price_deviation',
            'sender_tx_count', 'contract_age', 'is_contract_creation',
            'function_signature_hash', 'block_gas_used_ratio'
        ]
    
    def predict(self, features: dict) -> dict:
        """
        Predict if transaction is malicious
        
        Args:
            features: Dictionary of transaction features
        
        Returns:
            {
                'is_malicious': bool,
                'confidence': float,
                'probabilities': list
            }
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            # Predict
            probabilities = self.model.predict_proba(X_scaled)[0]
            prediction = self.model.predict(X_scaled)[0]
            
            return {
                'is_malicious': bool(prediction),
                'confidence': float(probabilities[1] if prediction else probabilities[0]),
                'probabilities': probabilities.tolist()
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
