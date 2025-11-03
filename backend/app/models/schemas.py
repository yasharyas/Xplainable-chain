from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class AnalyzeRequest(BaseModel):
    """Request model for transaction analysis"""
    tx_hash: str = Field(..., description="Transaction hash to analyze")
    network: str = Field(default="ethereum", description="Blockchain network")
    transaction_data: Optional[Dict] = Field(None, description="Optional transaction data (if not fetching from chain)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tx_hash": "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060",
                "network": "ethereum",
                "transaction_data": {
                    "amount": 1.5,
                    "gas_price": 50,
                    "gas_used": 50000,
                    "num_transfers": 2,
                    "unique_addresses": 3,
                    "time_of_day": 14,
                    "contract_interaction": True,
                    "sender_tx_count": 100,
                    "receiver_tx_count": 50
                }
            }
        }

class AnalyzeResponse(BaseModel):
    """Response model for transaction analysis"""
    tx_hash: str
    is_malicious: bool
    risk_score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    explanation: Dict
    ipfs_hash: str
    blockchain_hash: str
    features: Dict
    timestamp: Optional[str] = None

class VerifyResponse(BaseModel):
    """Response model for verification"""
    tx_hash: str
    exists: bool
    verified: bool
    ipfs_hash: Optional[str] = None
    risk_score: Optional[int] = None
    auditor: Optional[str] = None
    timestamp: Optional[int] = None

class AuditLogEntry(BaseModel):
    """Audit log entry model"""
    tx_hash: str
    risk_score: int
    is_malicious: bool
    analyzed_at: str
    blockchain_hash: str
    ipfs_hash: str
