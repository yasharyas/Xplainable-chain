"""
Generate sample transaction data for training the malicious transaction detector.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)

def generate_transaction_data(n_samples=10000):
    """
    Generate synthetic transaction data with both legitimate and malicious patterns.
    
    Features:
    - amount: Transaction value in ETH
    - gas_price: Gas price in Gwei
    - gas_used: Gas consumed
    - num_transfers: Number of token transfers
    - unique_addresses: Number of unique addresses interacted with
    - time_of_day: Hour of day (0-23)
    - contract_interaction: Boolean flag
    - sender_tx_count: Historical transaction count from sender
    - receiver_tx_count: Historical transaction count to receiver
    """
    
    # Calculate split between malicious and legitimate
    n_malicious = int(n_samples * 0.3)  # 30% malicious
    n_legitimate = n_samples - n_malicious
    
    # Generate legitimate transactions
    legitimate_data = {
        'amount': np.random.exponential(scale=0.5, size=n_legitimate),  # Most transactions are small
        'gas_price': np.random.normal(loc=50, scale=15, size=n_legitimate),  # Normal gas prices
        'gas_used': np.random.normal(loc=50000, scale=20000, size=n_legitimate),  # Normal gas usage
        'num_transfers': np.random.poisson(lam=1.5, size=n_legitimate),  # Few transfers
        'unique_addresses': np.random.poisson(lam=2, size=n_legitimate),  # Few addresses
        'time_of_day': np.random.randint(0, 24, size=n_legitimate),  # Distributed throughout day
        'contract_interaction': np.random.binomial(1, 0.4, size=n_legitimate),  # 40% interact with contracts
        'sender_tx_count': np.random.poisson(lam=50, size=n_legitimate),  # Established accounts
        'receiver_tx_count': np.random.poisson(lam=50, size=n_legitimate),
        'is_malicious': np.zeros(n_legitimate)
    }
    
    # Generate malicious transactions with suspicious patterns
    malicious_data = {
        'amount': np.concatenate([
            np.random.exponential(scale=2, size=n_malicious//2),  # Large amounts
            np.random.exponential(scale=0.01, size=n_malicious//2)  # Dust amounts
        ]),
        'gas_price': np.random.normal(loc=80, scale=30, size=n_malicious),  # Higher gas prices
        'gas_used': np.random.normal(loc=100000, scale=50000, size=n_malicious),  # Higher gas usage
        'num_transfers': np.random.poisson(lam=8, size=n_malicious),  # Many transfers (layering)
        'unique_addresses': np.random.poisson(lam=10, size=n_malicious),  # Many addresses
        'time_of_day': np.random.choice([2, 3, 4, 5], size=n_malicious),  # Unusual hours
        'contract_interaction': np.random.binomial(1, 0.8, size=n_malicious),  # High contract interaction
        'sender_tx_count': np.random.poisson(lam=5, size=n_malicious),  # New/suspicious accounts
        'receiver_tx_count': np.random.poisson(lam=5, size=n_malicious),
        'is_malicious': np.ones(n_malicious)
    }
    
    # Combine datasets
    df_legitimate = pd.DataFrame(legitimate_data)
    df_malicious = pd.DataFrame(malicious_data)
    df = pd.concat([df_legitimate, df_malicious], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure no negative values
    df['amount'] = df['amount'].abs()
    df['gas_price'] = df['gas_price'].abs()
    df['gas_used'] = df['gas_used'].abs()
    df['num_transfers'] = df['num_transfers'].abs()
    df['unique_addresses'] = df['unique_addresses'].abs()
    
    return df

def main():
    print("Generating sample transaction data...")
    
    # Generate data
    df = generate_transaction_data(n_samples=10000)
    
    # Save to CSV
    output_path = 'data/sample_transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nGenerated {len(df)} transactions")
    print(f"Distribution:")
    print(f"   - Legitimate: {len(df[df['is_malicious'] == 0])} ({len(df[df['is_malicious'] == 0])/len(df)*100:.1f}%)")
    print(f"   - Malicious:  {len(df[df['is_malicious'] == 1])} ({len(df[df['is_malicious'] == 1])/len(df)*100:.1f}%)")
    print(f"\nSaved to: {output_path}")
    
    # show what the data looks like
    print("\nFeature Statistics:")
    print(df.describe())
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_malicious': int(df['is_malicious'].sum()),
        'n_legitimate': int((df['is_malicious'] == 0).sum()),
        'features': list(df.columns),
        'feature_ranges': {
            col: {'min': float(df[col].min()), 'max': float(df[col].max()), 'mean': float(df[col].mean())}
            for col in df.columns if col != 'is_malicious'
        }
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nMetadata saved to: data/metadata.json")

if __name__ == '__main__':
    main()
