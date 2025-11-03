import requests
import json
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class IPFSService:
    """IPFS service using Pinata API"""
    
    def __init__(self):
        """Initialize Pinata API client"""
        self.api_key = os.getenv("PINATA_API_KEY")
        self.api_secret = os.getenv("PINATA_API_SECRET")
        self.base_url = "https://api.pinata.cloud"
        self.gateway_url = "https://gateway.pinata.cloud/ipfs"
    
    async def upload_json(self, data: Dict) -> str:
        """
        Upload JSON to IPFS via Pinata
        
        Args:
            data: Dictionary to upload
        
        Returns:
            IPFS CID (hash)
        """
        try:
            if not self.api_key or not self.api_secret or self.api_key == "":
                logger.warning("Pinata credentials not set, returning mock IPFS hash")
                return self._generate_mock_hash(data)
            
            headers = {
                "pinata_api_key": self.api_key,
                "pinata_secret_api_key": self.api_secret
            }
            
            url = f"{self.base_url}/pinning/pinJSONToIPFS"
            
            response = requests.post(
                url,
                json={"pinataContent": data},
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            ipfs_hash = result['IpfsHash']
            
            logger.info(f"Uploaded to IPFS: {ipfs_hash}")
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"IPFS upload error: {e}")
            logger.warning("Returning mock IPFS hash")
            return self._generate_mock_hash(data)
    
    def _generate_mock_hash(self, data: Dict) -> str:
        """Generate a mock IPFS hash for testing"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.sha256(data_str.encode())
        return "Qm" + hash_obj.hexdigest()[:44]
    
    async def get_json(self, ipfs_hash: str) -> Dict:
        """
        Retrieve JSON from IPFS
        
        Args:
            ipfs_hash: IPFS CID
        
        Returns:
            Retrieved JSON data
        """
        try:
            url = f"{self.gateway_url}/{ipfs_hash}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"IPFS retrieval error: {e}")
            raise
    
    def get_gateway_url(self, ipfs_hash: str) -> str:
        """Get the gateway URL for an IPFS hash"""
        return f"{self.gateway_url}/{ipfs_hash}"
