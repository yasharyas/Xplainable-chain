"""
Real-time Data Collection Pipeline
Fetches actual blockchain transactions and stores predictions in MongoDB
FIXES: Empty MongoDB database issue
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict
import os
from pymongo import MongoClient
from web3 import Web3
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionCollector:
    """Collect real Ethereum/Polygon transactions and store fraud predictions"""
    
    def __init__(self):
        # MongoDB connection (use Docker service name 'mongodb' not 'localhost')
        mongo_host = os.getenv('MONGO_HOST', 'mongodb')
        mongo_port = int(os.getenv('MONGO_PORT', 27017))
        self.mongo_client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
        self.db = self.mongo_client['xai_chain']
        self.collection = self.db['fraud_predictions']
        
        # Web3 connections
        self.infura_key = os.getenv('INFURA_API_KEY', 'your_infura_key')
        self.w3_eth = Web3(Web3.HTTPProvider(f'https://mainnet.infura.io/v3/{self.infura_key}'))
        self.w3_polygon = Web3(Web3.HTTPProvider(f'https://polygon-mainnet.infura.io/v3/{self.infura_key}'))
        
        # Etherscan API
        self.etherscan_key = os.getenv('ETHERSCAN_API_KEY', 'YourApiKeyToken')
        
        logger.info(" TransactionCollector initialized")
        
    async def collect_recent_transactions(self, network: str = 'ethereum', count: int = 100):
        """Fetch recent transactions from blockchain"""
        logger.info(f"Fetching {count} recent transactions from {network}...")
        
        if network == 'ethereum':
            return await self._fetch_ethereum_transactions(count)
        elif network == 'polygon':
            return await self._fetch_polygon_transactions(count)
        else:
            raise ValueError(f"Unknown network: {network}")
    
    async def _fetch_ethereum_transactions(self, count: int) -> List[Dict]:
        """Fetch from Ethereum mainnet"""
        transactions = []
        
        try:
            # Get latest block
            latest_block = self.w3_eth.eth.block_number
            
            # Fetch last N blocks
            for block_num in range(latest_block - count, latest_block):
                try:
                    block = self.w3_eth.eth.get_block(block_num, full_transactions=True)
                    
                    for tx in block.transactions[:5]:  # First 5 txs per block
                        tx_dict = {
                            'hash': tx['hash'].hex(),
                            'from': tx['from'],
                            'to': tx['to'] if tx['to'] else None,
                            'value': tx['value'],
                            'gas': tx['gas'],
                            'gasPrice': tx['gasPrice'],
                            'nonce': tx['nonce'],
                            'blockNumber': tx['blockNumber'],
                            'transactionIndex': tx['transactionIndex'],
                            'input': tx['input'].hex(),
                            'network': 'ethereum'
                        }
                        transactions.append(tx_dict)
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch block {block_num}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Ethereum fetch error: {e}")
        
        logger.info(f"Fetched {len(transactions)} Ethereum transactions")
        return transactions
    
    async def _fetch_polygon_transactions(self, count: int) -> List[Dict]:
        """Fetch from Polygon using Etherscan API"""
        url = 'https://api.polygonscan.com/api'
        
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': '0x7ceb23fd6bc0add59e62ac25578270cff1b9f619',  # Wrapped ETH on Polygon
            'startblock': 0,
            'endblock': 99999999,
            'page': 1,
            'offset': count,
            'sort': 'desc',
            'apikey': self.etherscan_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == '1':
                transactions = data.get('result', [])
                for tx in transactions:
                    tx['network'] = 'polygon'
                logger.info(f"Fetched {len(transactions)} Polygon transactions")
                return transactions
        except Exception as e:
            logger.error(f"Polygon fetch error: {e}")
        
        return []
    
    async def analyze_and_store_transaction(self, tx: Dict):
        """Analyze transaction and store prediction in MongoDB"""
        try:
            # Extract features
            features = self._extract_features(tx)
            
            # Run fraud detection (import here to avoid circular dependency)
            from app.models.ai_detector import AIDetector
            detector = AIDetector()
            
            prediction = detector.predict(features)
            
            # Prepare document
            document = {
                'transaction_hash': tx['hash'],
                'network': tx.get('network', 'ethereum'),
                'timestamp': datetime.utcnow(),
                'features': features,
                'prediction': prediction['is_malicious'],
                'fraud_probability': prediction['probabilities'][1],
                'confidence': prediction['confidence'],
                'from_address': tx.get('from'),
                'to_address': tx.get('to'),
                'value': int(tx.get('value', 0)),
                'gas_price': int(tx.get('gasPrice', 0)),
                'gas_used': int(tx.get('gas', 0)),
            }
            
            # Check if already exists
            existing = self.collection.find_one({'transaction_hash': tx['hash']})
            
            if not existing:
                self.collection.insert_one(document)
                logger.debug(f" Stored prediction for {tx['hash'][:10]}...")
                return True
            else:
                logger.debug(f"⏭️  Skipped duplicate {tx['hash'][:10]}...")
                return False
                
        except Exception as e:
            logger.error(f"Failed to analyze transaction: {e}")
            return False
    
    def _extract_features(self, tx: Dict) -> Dict[str, float]:
        """Extract ML features from transaction"""
        value_wei = int(tx.get('value', 0))
        gas_price = int(tx.get('gasPrice', 0))
        gas_used = int(tx.get('gas', 0))
        
        return {
            'amount': value_wei / 1e18,
            'gas_price': gas_price / 1e9,
            'gas_used': gas_used,
            'gas_price_deviation': abs(gas_price / 1e9 - 50) / 50,
            'value': value_wei / 1e18,
            'sender_tx_count': int(tx.get('nonce', 0)),
            'is_contract_creation': 1 if not tx.get('to') else 0,
            'contract_age': 0,  # Would need additional API calls
            'block_gas_used_ratio': 0.5,  # Placeholder
        }
    
    async def populate_database(self, target_count: int = 500):
        """Populate MongoDB with fraud predictions"""
        logger.info("=" * 60)
        logger.info(f"POPULATING DATABASE WITH {target_count} TRANSACTIONS")
        logger.info("=" * 60)
        
        current_count = self.collection.count_documents({})
        logger.info(f"Current database size: {current_count} documents")
        
        if current_count >= target_count:
            logger.info(" Database already populated")
            return
        
        needed = target_count - current_count
        logger.info(f"Need {needed} more transactions")
        
        # Fetch from multiple networks
        eth_txs = await self.collect_recent_transactions('ethereum', needed // 2)
        
        # Try Polygon if Ethereum fails
        if len(eth_txs) < needed // 2:
            poly_txs = await self.collect_recent_transactions('polygon', needed // 2)
            all_txs = eth_txs + poly_txs
        else:
            all_txs = eth_txs
        
        logger.info(f"Collected {len(all_txs)} transactions total")
        
        # Analyze and store
        stored_count = 0
        for i, tx in enumerate(all_txs[:needed]):
            success = await self.analyze_and_store_transaction(tx)
            if success:
                stored_count += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(all_txs[:needed])} analyzed, {stored_count} stored")
        
        final_count = self.collection.count_documents({})
        logger.info("=" * 60)
        logger.info(f" DATABASE POPULATION COMPLETE")
        logger.info(f"Total documents: {final_count}")
        logger.info(f"New documents added: {stored_count}")
        logger.info("=" * 60)
    
    def get_statistics(self):
        """Get database statistics"""
        total = self.collection.count_documents({})
        fraud = self.collection.count_documents({'prediction': True})
        legit = total - fraud
        
        logger.info(f"\nDatabase Statistics:")
        logger.info(f"  Total: {total}")
        logger.info(f"  Fraud: {fraud} ({fraud/max(total,1)*100:.1f}%)")
        logger.info(f"  Legitimate: {legit} ({legit/max(total,1)*100:.1f}%)")
        
        return {'total': total, 'fraud': fraud, 'legitimate': legit}


async def main():
    """Main data collection pipeline"""
    collector = TransactionCollector()
    
    # Populate database
    await collector.populate_database(target_count=500)
    
    # Show stats
    collector.get_statistics()


if __name__ == '__main__':
    asyncio.run(main())
