"""
Add Synthetic Fraud Cases to MongoDB
ROOT ISSUE: Real blockchain data has only 2 fraud cases (0.4%)
SOLUTION: Add 150 synthetic fraud transactions with realistic patterns
"""

import os
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_fraud_transaction():
    """Generate a realistic fraudulent transaction"""
    # High gas price (front-running)
    if np.random.random() < 0.4:
        gas_price = np.random.uniform(180, 350)
        gas_price_deviation = abs(gas_price - 50) / 50
        pattern = "front-running"
    # Phishing with new account
    elif np.random.random() < 0.3:
        gas_price = np.random.uniform(40, 80)
        gas_price_deviation = abs(gas_price - 50) / 50
        pattern = "phishing"
    # Contract exploit
    else:
        gas_price = np.random.uniform(100, 200)
        gas_price_deviation = abs(gas_price - 50) / 50
        pattern = "exploit"
    
    # Generate features based on pattern
    if pattern == "front-running":
        features = {
            'amount': np.random.uniform(0.1, 20),
            'gas_price': gas_price,
            'gas_used': np.random.uniform(80000, 200000),
            'gas_price_deviation': gas_price_deviation,
            'value': np.random.uniform(100000000, 20000000000),
            'sender_tx_count': np.random.randint(50, 500),  # Experienced account
            'is_contract_creation': 0,
            'contract_age': 0,
            'block_gas_used_ratio': np.random.uniform(0.6, 0.95)
        }
        fraud_prob = np.random.uniform(0.75, 0.95)
        
    elif pattern == "phishing":
        features = {
            'amount': np.random.uniform(5, 50),
            'gas_price': gas_price,
            'gas_used': np.random.uniform(21000, 60000),
            'gas_price_deviation': gas_price_deviation,
            'value': np.random.uniform(5000000000, 50000000000),  # Large value
            'sender_tx_count': np.random.randint(1, 15),  # New account
            'is_contract_creation': 0,
            'contract_age': 0,
            'block_gas_used_ratio': np.random.uniform(0.3, 0.7)
        }
        fraud_prob = np.random.uniform(0.65, 0.85)
        
    else:  # exploit
        features = {
            'amount': np.random.uniform(1, 15),
            'gas_price': gas_price,
            'gas_used': np.random.uniform(100000, 300000),
            'gas_price_deviation': gas_price_deviation,
            'value': np.random.uniform(1000000000, 15000000000),
            'sender_tx_count': np.random.randint(5, 50),  # Moderately new
            'is_contract_creation': np.random.choice([0, 1], p=[0.3, 0.7]),  # Often contract
            'contract_age': np.random.randint(0, 30),
            'block_gas_used_ratio': np.random.uniform(0.5, 0.9)
        }
        fraud_prob = np.random.uniform(0.70, 0.90)
    
    risk_score = int(fraud_prob * 100)
    
    # Convert numpy types to Python natives
    features_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in features.items()}
    
    return {
        'tx_hash': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
        'is_malicious': True,
        'risk_score': int(risk_score),
        'fraud_probability': float(fraud_prob),
        'confidence': float(fraud_prob),
        'features': features_clean,
        'pattern': pattern,
        'synthetic': True,  # Mark as synthetic
        'timestamp': datetime.now().isoformat()
    }


def generate_legitimate_transaction():
    """Generate a realistic legitimate transaction"""
    gas_price = np.random.uniform(30, 70)
    
    features = {
        'amount': np.random.exponential(0.5),  # Small amounts
        'gas_price': gas_price,
        'gas_used': np.random.uniform(21000, 80000),
        'gas_price_deviation': abs(gas_price - 50) / 50,
        'value': np.random.uniform(10000000, 5000000000),
        'sender_tx_count': np.random.randint(20, 5000),  # Established accounts
        'is_contract_creation': np.random.choice([0, 1], p=[0.9, 0.1]),
        'contract_age': np.random.randint(30, 1000),
        'block_gas_used_ratio': np.random.uniform(0.3, 0.7)
    }
    
    fraud_prob = np.random.uniform(0.05, 0.35)
    risk_score = int(fraud_prob * 100)
    
    # Convert numpy types to Python natives
    features_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in features.items()}
    
    return {
        'tx_hash': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
        'is_malicious': False,
        'risk_score': int(risk_score),
        'fraud_probability': float(fraud_prob),
        'confidence': float(1 - fraud_prob),
        'features': features_clean,
        'pattern': 'legitimate',
        'synthetic': True,
        'timestamp': datetime.now().isoformat()
    }


def add_synthetic_fraud_to_mongodb(n_fraud=150, n_legit=50):
    """Add synthetic transactions to MongoDB for causal discovery"""
    
    mongo_host = os.getenv('MONGO_HOST', 'mongodb')
    mongo_port = int(os.getenv('MONGO_PORT', 27017))
    client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
    
    db = client['xai_chain']
    collection = db['fraud_predictions']
    
    logger.info("="*70)
    logger.info("ADDING SYNTHETIC FRAUD CASES FOR CAUSAL DISCOVERY")
    logger.info("="*70)
    
    # Generate fraud transactions
    logger.info(f"\nGenerating {n_fraud} fraud transactions...")
    fraud_docs = [generate_fraud_transaction() for _ in range(n_fraud)]
    
    # Generate legitimate transactions
    logger.info(f"Generating {n_legit} legitimate transactions...")
    legit_docs = [generate_legitimate_transaction() for _ in range(n_legit)]
    
    # Insert into MongoDB
    all_docs = fraud_docs + legit_docs
    logger.info(f"\nInserting {len(all_docs)} transactions into MongoDB...")
    result = collection.insert_many(all_docs)
    
    logger.info(f" Inserted {len(result.inserted_ids)} documents")
    
    # Verify final distribution
    total = collection.count_documents({})
    fraud_count = collection.count_documents({'is_malicious': True})
    synthetic_count = collection.count_documents({'synthetic': True})
    
    logger.info(f"\n{'='*70}")
    logger.info("FINAL MONGODB STATISTICS")
    logger.info(f"{'='*70}")
    logger.info(f"Total transactions: {total}")
    logger.info(f"Fraud cases: {fraud_count} ({fraud_count/total*100:.1f}%)")
    logger.info(f"Legitimate cases: {total - fraud_count} ({(total - fraud_count)/total*100:.1f}%)")
    logger.info(f"Synthetic transactions: {synthetic_count} ({synthetic_count/total*100:.1f}%)")
    logger.info(f"Real transactions: {total - synthetic_count} ({(total - synthetic_count)/total*100:.1f}%)")
    
    # Check variance
    docs = list(collection.find({}, {'risk_score': 1}))
    risk_scores = [d.get('risk_score', 0) for d in docs]
    
    logger.info(f"\nRisk score distribution:")
    logger.info(f"  Range: [{min(risk_scores)}, {max(risk_scores)}]")
    logger.info(f"  Mean: {np.mean(risk_scores):.2f}")
    logger.info(f"  Std: {np.std(risk_scores):.2f}")
    logger.info(f"  Variance: {np.var(risk_scores):.2f}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    add_synthetic_fraud_to_mongodb(n_fraud=150, n_legit=50)
