import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "XAI-Chain API"
    API_VERSION: str = "1.0.0"
    
    # Blockchain
    INFURA_URL: str = os.getenv("INFURA_URL", "https://polygon-mumbai.infura.io/v3/YOUR_KEY")
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
    
    # IPFS
    PINATA_API_KEY: str = os.getenv("PINATA_API_KEY", "")
    PINATA_API_SECRET: str = os.getenv("PINATA_API_SECRET", "")
    PINATA_JWT: str = os.getenv("PINATA_JWT", "")
    
    # Database
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/xaichain")
    
    # APIs
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
    
    # ML Model Paths
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "ml", "model.pkl")
    SCALER_PATH: str = os.path.join(os.path.dirname(__file__), "..", "ml", "scaler.pkl")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
