from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, List
import logging

from app.routers import analyze, verify, audit
from app.utils.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XAI-Chain API",
    description="Explainable AI for Blockchain Security",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://xai-chain.vercel.app",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze.router, prefix="/api/analyze", tags=["analyze"])
app.include_router(verify.router, prefix="/api/verify", tags=["verify"])
app.include_router(audit.router, prefix="/api/audit", tags=["audit"])

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "XAI-Chain API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/analyze",
            "verify": "/api/verify/{tx_hash}",
            "audit": "/api/audit",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
