from web3 import Web3
from eth_account import Account
import json
import os
from typing import Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BlockchainService:
    """Web3 service for blockchain interactions"""
    
    def __init__(self):
        """Initialize Web3 connection"""
        # Connect to Infura or other RPC
        infura_url = os.getenv("INFURA_URL", "https://polygon-mumbai.infura.io/v3/YOUR_KEY")
        self.w3 = Web3(Web3.HTTPProvider(infura_url))
        
        # Load contract
        contract_address = os.getenv("CONTRACT_ADDRESS")
        
        # Try to load contract ABI
        abi_path = Path(__file__).parent.parent.parent.parent / "frontend" / "public" / "abi" / "XAIChainVerifier.json"
        
        try:
            if abi_path.exists():
                with open(abi_path, "r") as f:
                    contract_data = json.load(f)
                    contract_abi = contract_data.get('abi', [])
            else:
                logger.warning(f"Contract ABI not found at {abi_path}")
                contract_abi = self._get_mock_abi()
            
            if contract_address and contract_address != "":
                self.contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(contract_address),
                    abi=contract_abi
                )
            else:
                logger.warning("Contract address not set. Blockchain storage will be mocked.")
                self.contract = None
            
            # Load private key for transactions
            private_key = os.getenv("PRIVATE_KEY")
            if private_key and private_key != "":
                self.account = Account.from_key(private_key)
            else:
                logger.warning("Private key not set. Transactions will be mocked.")
                self.account = None
            
            logger.info(f"Blockchain service initialized. Connected: {self.w3.is_connected()}")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain service: {e}")
            self.contract = None
            self.account = None
    
    def _get_mock_abi(self):
        """Return a mock ABI for testing"""
        return [
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "_txHash", "type": "bytes32"},
                    {"internalType": "string", "name": "_ipfsHash", "type": "string"},
                    {"internalType": "uint8", "name": "_riskScore", "type": "uint8"}
                ],
                "name": "storeExplanation",
                "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    async def get_transaction(self, tx_hash: str, network: str = "ethereum") -> Optional[Dict]:
        """
        Fetch transaction data from blockchain
        
        Args:
            tx_hash: Transaction hash
            network: Blockchain network
        
        Returns:
            Transaction data dictionary
        """
        try:
            if not self.w3.is_connected():
                logger.warning("Web3 not connected, returning mock data")
                return self._get_mock_transaction(tx_hash)
            
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            block = self.w3.eth.get_block(tx['blockNumber'])
            
            return {
                'hash': tx_hash,
                'from': tx['from'],
                'to': tx['to'] if tx['to'] else '',
                'value': float(self.w3.from_wei(tx['value'], 'ether')),
                'gas': tx['gas'],
                'gas_price': float(self.w3.from_wei(tx['gasPrice'], 'gwei')),
                'gas_used': receipt['gasUsed'],
                'block_number': tx['blockNumber'],
                'timestamp': block['timestamp'],
                'input': tx['input'].hex() if hasattr(tx['input'], 'hex') else tx['input']
            }
        except Exception as e:
            logger.error(f"Error fetching transaction: {e}")
            return self._get_mock_transaction(tx_hash)
    
    def _get_mock_transaction(self, tx_hash: str) -> Dict:
        """Return mock transaction data for testing"""
        import random
        return {
            'hash': tx_hash,
            'from': '0x' + '1' * 40,
            'to': '0x' + '2' * 40,
            'value': random.uniform(0.1, 10.0),
            'gas': random.randint(21000, 100000),
            'gas_price': random.uniform(20, 100),
            'gas_used': random.randint(21000, 90000),
            'block_number': random.randint(1000000, 2000000),
            'timestamp': 1609459200,
            'input': '0x'
        }
    
    async def store_explanation(
        self,
        tx_hash: str,
        ipfs_hash: str,
        risk_score: int
    ) -> str:
        """
        Store explanation on blockchain
        
        Args:
            tx_hash: Transaction hash being analyzed
            ipfs_hash: IPFS CID of explanation
            risk_score: Risk score 0-100
        
        Returns:
            Transaction hash of storage transaction
        """
        try:
            if not self.contract or not self.account:
                logger.warning("Contract or account not available, returning mock hash")
                return "0x" + "mock_blockchain_hash_" + tx_hash[2:20]
            
            # Convert tx_hash to bytes32
            tx_hash_bytes = Web3.to_bytes(hexstr=tx_hash)
            
            # Build transaction
            txn = self.contract.functions.storeExplanation(
                tx_hash_bytes,
                ipfs_hash,
                risk_score
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send
            signed_txn = self.account.sign_transaction(txn)
            tx_hash_result = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_result)
            
            result_hash = receipt['transactionHash'].hex()
            logger.info(f"Stored on blockchain: {result_hash}")
            return result_hash
            
        except Exception as e:
            logger.error(f"Blockchain storage error: {e}")
            logger.warning("Returning mock blockchain hash")
            return "0x" + "mock_blockchain_hash_" + tx_hash[2:20]
