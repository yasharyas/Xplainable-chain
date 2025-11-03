"""
Lightweight backend startup - Optimized for minimal resource usage
Uses production settings with reduced workers and no hot reload
"""
import os
os.environ['NUMEXPR_MAX_THREADS'] = '2'  # Limit numpy threads
os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.routers import analyze, verify, audit

# Lightweight FastAPI configuration
app = FastAPI(
    title="XAI-Chain API",
    description="Explainable AI for Blockchain Security",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,  # Disable ReDoc to save resources
    openapi_url="/openapi.json"
)

# Minimal CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze.router, prefix="/api/analyze", tags=["analyze"])
app.include_router(verify.router, prefix="/api/verify", tags=["verify"])
app.include_router(audit.router, prefix="/api/audit", tags=["audit"])

@app.get("/")
def root():
    return {
        "message": "XAI-Chain API (Lightweight Mode)",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Production-optimized settings
    uvicorn.run(
        "main_lightweight:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker
        reload=False,  # No hot reload
        log_level="warning",  # Reduced logging
        access_log=False  # Disable access logs
    )
