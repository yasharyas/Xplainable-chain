import requests
import json
import os
from typing import Dict
import logging
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class IPFSService:
    """IPFS service using Pinata API"""
    
    def __init__(self):
        """Initialize Pinata API client"""
        self.api_key = os.getenv("PINATA_API_KEY")
        self.api_secret = os.getenv("PINATA_API_SECRET")
        self.jwt = os.getenv("PINATA_JWT")
        self.base_url = "https://api.pinata.cloud"
        
        # Use JWT if available (recommended), otherwise use API key/secret
        self.use_jwt = bool(self.jwt)
        logger.info(f"IPFSService initialized (Mode: {'JWT' if self.use_jwt else 'API Key' if self.api_key else 'Mock'})")
        self.gateway_url = "https://gateway.pinata.cloud/ipfs"
    
    async def upload_json(self, data: Dict) -> str:
        """
        Upload JSON to IPFS via Pinata
        
        Returns:
            IPFS CID (hash)
            
        Raises:
            HTTPException: If Pinata credentials are not configured or upload fails
        """
        # Check if API credentials are configured
        if not (self.jwt or (self.api_key and self.api_secret)):
            logger.error("IPFS upload FAILED: Pinata API credentials not configured in .env")
            raise HTTPException(
                status_code=503,
                detail="IPFS service not configured. Please set PINATA_JWT in environment variables."
            )
        
        try:
            # Prepare headers (prefer JWT)
            if self.use_jwt:
                headers = {
                    "Authorization": f"Bearer {self.jwt}",
                    "Content-Type": "application/json"
                }
            else:
                headers = {
                    "pinata_api_key": self.api_key,
                    "pinata_secret_api_key": self.api_secret,
                    "Content-Type": "application/json"
                }
            
            url = f"{self.base_url}/pinning/pinJSONToIPFS"
            
            # Create the payload
            payload = {
                "pinataContent": data,
                "pinataMetadata": {
                    "name": f"XAI-Chain-{data.get('tx_hash', 'analysis')}"
                }
            }
            
            logger.info(f"Uploading to IPFS via Pinata...")
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            ipfs_hash = result['IpfsHash']
            
            logger.info(f"✅ Successfully uploaded to IPFS: {ipfs_hash}")
            logger.info(f"📎 View at: https://gateway.pinata.cloud/ipfs/{ipfs_hash}")
            return ipfs_hash
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"IPFS HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to upload to IPFS: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            logger.error(f"IPFS upload error: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"IPFS upload failed: {str(e)}"
            )
    
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
