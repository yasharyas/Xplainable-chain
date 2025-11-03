import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_transactions(n=10000):
    """
    Generate synthetic transaction data for testing
    
    Args:
        n: Number of transactions to generate
    
    Returns:
        pandas DataFrame with transaction data
    """
    np.random.seed(42)
    
    transactions = []
    base_time = int(datetime.now().timestamp())
    
    print(f"Generating {n} sample transactions...")
    
    for i in range(n):
        # 90% normal, 10% malicious
        is_malicious = 1 if np.random.random() < 0.1 else 0
        
        if is_malicious:
            # Malicious patterns
            gas_price = np.random.uniform(80, 200)  # High gas (front-running)
            gas_used = np.random.randint(100000, 500000)
            value = np.random.uniform(50, 200)  # Large transfers
            sender_tx_count = np.random.randint(1, 50)  # New address
            contract_age = np.random.randint(1, 30)  # New contract
        else:
            # Normal patterns
            gas_price = np.random.uniform(20, 60)
            gas_used = np.random.randint(21000, 100000)
            value = np.random.uniform(0, 10)
            sender_tx_count = np.random.randint(50, 1000)
            contract_age = np.random.randint(30, 1000)
        
        tx = {
            'hash': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
            'from': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
            'to': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
            'value': value,
            'gas': gas_used,
            'gas_price': gas_price,
            'gas_used': gas_used,
            'block_number': 10000000 + i,
            'timestamp': base_time - (i * 15),  # 15 sec per block
            'is_malicious': is_malicious,
            # Features
            'gas_price_deviation': abs(gas_price - 40) / 40,
            'sender_tx_count': sender_tx_count,
            'contract_age': contract_age,
            'is_contract_creation': np.random.choice([0, 1], p=[0.95, 0.05]),
            'function_signature_hash': np.random.randint(0, 10000),
            'block_gas_used_ratio': np.random.uniform(0.3, 0.9)
        }
        transactions.append(tx)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n} transactions...")
    
    df = pd.DataFrame(transactions)
    return df

if __name__ == "__main__":
    # Generate training data
    print("Starting data generation...")
    df = generate_sample_transactions(10000)
    
    # Save to CSV
    output_path = 'data/processed/features.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(df)} sample transactions")
    print(f"💾 Saved to: {output_path}")
    print(f"\nData distribution:")
    print(f"  Malicious: {df['is_malicious'].sum()} ({df['is_malicious'].sum() / len(df) * 100:.1f}%)")
    print(f"  Normal: {(df['is_malicious'] == 0).sum()} ({(df['is_malicious'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"\nSample statistics:")
    print(df[['gas_price', 'value', 'gas_price_deviation', 'sender_tx_count']].describe())
