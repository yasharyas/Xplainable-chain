import numpy as np
from typing import Dict
from web3 import Web3

def extract_features(tx_data: Dict) -> Dict:
    """
    Extract ML features from transaction data
    IMPORTANT: Must match features used during model training!
    
    Args:
        tx_data: Raw transaction data from blockchain
    
    Returns:
        Dictionary of engineered features matching training data
    """
    features = {}
    
    # Feature 1: amount (transaction value in ETH)
    features['amount'] = float(tx_data.get('value', 0) / 1e18) if tx_data.get('value') else 0.5
    
    # Feature 2: gas_price (in Gwei)
    features['gas_price'] = float(tx_data.get('gas_price', 50))
    
    # Feature 3: gas_used
    features['gas_used'] = int(tx_data.get('gas_used', 50000))
    
    # Feature 4: num_transfers (number of token transfers in transaction)
    features['num_transfers'] = int(tx_data.get('num_transfers', 1))
    
    # Feature 5: unique_addresses (number of unique addresses involved)
    features['unique_addresses'] = int(tx_data.get('unique_addresses', 2))
    
    # Feature 6: time_of_day (0-23)
    from datetime import datetime
    features['time_of_day'] = int(tx_data.get('time_of_day', datetime.now().hour))
    
    # Feature 7: contract_interaction (boolean: 1 or 0)
    to_address = tx_data.get('to', '')
    input_data = tx_data.get('input', '0x')
    features['contract_interaction'] = 1 if (input_data and input_data != '0x') else 0
    
    # Feature 8: sender_tx_count (historical transactions from sender)
    features['sender_tx_count'] = int(tx_data.get('sender_tx_count', 50))
    
    # Feature 9: receiver_tx_count (historical transactions to receiver)
    features['receiver_tx_count'] = int(tx_data.get('receiver_tx_count', 50))
    
    return features

def get_sender_history(address: str, web3_instance=None) -> int:
    """Get number of transactions from address (simplified for MVP)"""
    # TODO: Query Etherscan API or database
    # For MVP: return random value between 1-1000
    return np.random.randint(1, 1000)

def calculate_gas_deviation(gas_price: float, web3_instance=None) -> float:
    """Calculate deviation from average gas price (simplified for MVP)"""
    # TODO: Query recent blocks for average
    # For MVP: use fixed average
    avg_gas = 50.0
    if avg_gas > 0:
        return abs(gas_price - avg_gas) / avg_gas
    return 0.0

def get_contract_age(address: str, web3_instance=None) -> int:
    """Get contract age in days (simplified for MVP)"""
    # TODO: Query Etherscan for contract creation
    # For MVP: return random value
    return np.random.randint(1, 1000)
