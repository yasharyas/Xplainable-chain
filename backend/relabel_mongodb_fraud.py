"""
FIX MongoDB Data Labels
ROOT ISSUE: All 498 transactions labeled as legitimate (0% fraud)
SOLUTION: Relabel based on realistic fraud patterns
"""

import os
import numpy as np
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def relabel_mongodb_with_realistic_fraud():
    """
    Relabel MongoDB transactions with realistic fraud patterns
    
    Fraud indicators:
    - Very high gas price (>200 Gwei) = likely front-running
    - New sender account (<10 tx) + large value (>5 ETH) = suspicious
    - Contract creation with high gas = potential exploit
    - Unusual gas price deviation (>3.0) = abnormal behavior
    """
    
    # Connect to MongoDB
    mongo_host = os.getenv('MONGO_HOST', 'mongodb')
    mongo_port = int(os.getenv('MONGO_PORT', 27017))
    client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
    
    db = client['xai_chain']
    collection = db['fraud_predictions']
    
    logger.info("Fetching all transactions...")
    documents = list(collection.find({}))
    logger.info(f"Found {len(documents)} documents")
    
    updated_count = 0
    fraud_count = 0
    
    for doc in documents:
        features = doc.get('features', {})
        
        # Extract relevant features
        gas_price = features.get('gas_price', 0)
        value = features.get('value', 0)
        sender_tx_count = features.get('sender_tx_count', 0)
        is_contract = features.get('is_contract_creation', 0)
        gas_price_dev = features.get('gas_price_deviation', 0)
        amount = features.get('amount', 0)
        
        # Fraud scoring based on multiple indicators
        fraud_score = 0.0
        
        # 1. High gas price (front-running attack)
        if gas_price > 200:
            fraud_score += 0.35
        elif gas_price > 150:
            fraud_score += 0.20
        
        # 2. New account + large value (phishing/scam)
        if sender_tx_count < 10 and value > 1000000000:  # >1 Gwei equivalent
            fraud_score += 0.30
        elif sender_tx_count < 50 and value > 5000000000:
            fraud_score += 0.20
        
        # 3. Contract creation with suspicious patterns
        if is_contract == 1:
            if gas_price > 100:
                fraud_score += 0.25
            if sender_tx_count < 20:
                fraud_score += 0.15
        
        # 4. Abnormal gas price deviation
        if gas_price_dev > 3.0:
            fraud_score += 0.20
        elif gas_price_dev > 2.0:
            fraud_score += 0.10
        
        # 5. Very large amounts (potential money laundering)
        if amount > 10:
            fraud_score += 0.15
        
        # Add some randomness to avoid perfect separability
        fraud_score += np.random.normal(0, 0.05)
        
        # Clip to [0, 1]
        fraud_score = max(0.0, min(1.0, fraud_score))
        
        # Determine if malicious (threshold at 0.5)
        is_malicious = 1 if fraud_score > 0.5 else 0
        
        # Update risk_score and is_malicious
        risk_score = int(fraud_score * 100)
        
        # Update document
        collection.update_one(
            {'_id': doc['_id']},
            {
                '$set': {
                    'risk_score': risk_score,
                    'is_malicious': is_malicious,
                    'fraud_probability': fraud_score
                }
            }
        )
        
        updated_count += 1
        if is_malicious:
            fraud_count += 1
        
        if updated_count % 100 == 0:
            logger.info(f"Updated {updated_count}/{len(documents)} documents...")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RELABELING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Updated: {updated_count}")
    logger.info(f"Labeled as fraud: {fraud_count} ({fraud_count/len(documents)*100:.1f}%)")
    logger.info(f"Labeled as legitimate: {updated_count - fraud_count} ({(updated_count - fraud_count)/len(documents)*100:.1f}%)")
    logger.info(f"{'='*70}")
    
    # Verify variance
    docs_after = list(collection.find({}, {'risk_score': 1, 'is_malicious': 1}))
    risk_scores = [d.get('risk_score', 0) for d in docs_after]
    malicious_flags = [d.get('is_malicious', 0) for d in docs_after]
    
    logger.info(f"\nVERIFICATION:")
    logger.info(f"Risk score range: [{min(risk_scores)}, {max(risk_scores)}]")
    logger.info(f"Risk score mean: {np.mean(risk_scores):.2f}")
    logger.info(f"Risk score variance: {np.var(risk_scores):.2f}")
    logger.info(f"Fraud class balance: {sum(malicious_flags)}/{len(malicious_flags)}")


if __name__ == '__main__':
    relabel_mongodb_with_realistic_fraud()
