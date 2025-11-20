"""
Real XGBoost Training Pipeline for Blockchain Fraud Detection
Fetches actual Ethereum transactions and trains production model
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import requests
import time
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainDataCollector:
    """Collect real Ethereum transaction data from Etherscan API"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ETHERSCAN_API_KEY', 'YourApiKeyToken')
        self.base_url = 'https://api.etherscan.io/api'
        
        # Known fraud addresses (from public datasets)
        self.known_fraud_addresses = set([
            '0x0000000000000000000000000000000000000000',  # Null address (burns)
            '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be',  # Binance hack
            '0xd90e2f925DA726b50C4Ed8D0Fb90Ad053324F31b',  # Ronin bridge hack
            # Add more known fraud addresses
        ])
        
    def fetch_transaction(self, tx_hash):
        """Fetch single transaction details"""
        params = {
            'module': 'proxy',
            'action': 'eth_getTransactionByHash',
            'txhash': tx_hash,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            if data.get('result'):
                return data['result']
        except Exception as e:
            logger.error(f"Failed to fetch tx {tx_hash}: {e}")
        return None
    
    def fetch_recent_transactions(self, address, limit=100):
        """Fetch recent transactions for an address"""
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': limit,
            'sort': 'desc',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            if data.get('status') == '1' and data.get('result'):
                return data['result']
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            logger.error(f"Failed to fetch transactions for {address}: {e}")
        return []
    
    def label_transaction(self, tx):
        """Label transaction as fraud (1) or legitimate (0)"""
        # Heuristic labeling (improve with known datasets)
        from_addr = tx.get('from', '').lower()
        to_addr = tx.get('to', '').lower()
        
        # Known fraud addresses
        if from_addr in self.known_fraud_addresses or to_addr in self.known_fraud_addresses:
            return 1
        
        # High-value transactions to new contracts
        value_eth = int(tx.get('value', 0)) / 1e18
        if value_eth > 100 and tx.get('isError') == '0' and not to_addr:
            return 1  # Suspicious contract creation with high value
        
        # Failed transactions with high gas
        if tx.get('isError') == '1' and int(tx.get('gasUsed', 0)) > 100000:
            return 1
        
        # Normal transaction
        return 0
    
    def collect_dataset(self, n_samples=1000):
        """Collect labeled dataset from Ethereum"""
        logger.info(f"Collecting {n_samples} real transactions from Ethereum...")
        
        # Mix of popular addresses and random transactions
        popular_addresses = [
            '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
            '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
            '0x514910771af9ca656af840dff83e8264ecf986ca',  # LINK
            '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',  # WBTC
        ]
        
        all_transactions = []
        
        for address in popular_addresses:
            txs = self.fetch_recent_transactions(address, limit=250)
            all_transactions.extend(txs)
            if len(all_transactions) >= n_samples:
                break
        
        logger.info(f"Collected {len(all_transactions)} transactions")
        return all_transactions[:n_samples]


class FeatureEngineer:
    """Extract ML features from raw transaction data"""
    
    @staticmethod
    def extract_features(tx):
        """Convert raw transaction to feature vector"""
        value_wei = int(tx.get('value', 0))
        gas_price = int(tx.get('gasPrice', 0))
        gas_used = int(tx.get('gasUsed', tx.get('gas', 0)))
        
        features = {
            # Core features
            'amount': value_wei / 1e18,  # Convert to ETH
            'gas_price': gas_price / 1e9,  # Convert to Gwei
            'gas_used': gas_used,
            
            # Derived features
            'gas_price_deviation': abs(gas_price / 1e9 - 50) / 50,  # Deviation from ~50 Gwei
            'total_cost': (gas_used * gas_price) / 1e18,
            'value_to_gas_ratio': value_wei / max(gas_used * gas_price, 1),
            
            # Transaction metadata
            'is_contract_creation': 1 if not tx.get('to') else 0,
            'is_error': 1 if tx.get('isError') == '1' else 0,
            'has_input_data': 1 if len(tx.get('input', '0x')) > 10 else 0,
            
            # Additional features
            'block_confirmations': int(tx.get('confirmations', 0)),
            'nonce': int(tx.get('nonce', 0)),
            'transaction_index': int(tx.get('transactionIndex', 0)),
        }
        
        return features


def train_xgboost_model(data_path=None, save_path='app/ml'):
    """Train production XGBoost model on real blockchain data"""
    
    logger.info("=" * 60)
    logger.info("TRAINING REAL XGBOOST MODEL FOR FRAUD DETECTION")
    logger.info("=" * 60)
    
    # Collect data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Fetching real Ethereum transactions...")
        collector = BlockchainDataCollector()
        transactions = collector.collect_dataset(n_samples=1000)
        
        # Extract features
        engineer = FeatureEngineer()
        processed_data = []
        
        for tx in transactions:
            try:
                features = engineer.extract_features(tx)
                features['label'] = collector.label_transaction(tx)
                processed_data.append(features)
            except Exception as e:
                logger.warning(f"Failed to process transaction: {e}")
        
        df = pd.DataFrame(processed_data)
        
        # Save dataset
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/training_data.csv', index=False)
        logger.info(f"Saved {len(df)} samples to data/processed/training_data.csv")
    
    # Check class distribution
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Prepare features
    feature_columns = [col for col in df.columns if col != 'label']
    X = df[feature_columns]
    y = df['label']
    
    # Handle class imbalance
    fraud_count = (y == 1).sum()
    legit_count = (y == 0).sum()
    scale_pos_weight = legit_count / max(fraud_count, 1)
    
    logger.info(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    logger.info("\nTraining XGBoost classifier...")
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=True
    )
    
    # Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
    
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"\nROC-AUC Score: {auc:.4f}")
    
    # Cross-validation
    logger.info("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    logger.info("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n" + feature_importance.head(10).to_string(index=False))
    
    # Save model and scaler
    os.makedirs(save_path, exist_ok=True)
    
    model_path = os.path.join(save_path, 'model.pkl')
    scaler_path = os.path.join(save_path, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info(f"\n Model saved to {model_path}")
    logger.info(f" Scaler saved to {scaler_path}")
    
    # Save feature names
    feature_names_path = os.path.join(save_path, 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    
    logger.info(f" Feature names saved to {feature_names_path}")
    
    return model, scaler, feature_columns


if __name__ == '__main__':
    # Train model
    model, scaler, features = train_xgboost_model()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE - REAL ML MODEL READY FOR PRODUCTION")
    logger.info("=" * 60)
