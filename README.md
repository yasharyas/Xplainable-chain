# XAI-Chain: Explainable AI Blockchain Transaction Analyzer# XAI-Chain: Explainable AI for Blockchain Security



Full-stack platform using Machine Learning to detect malicious blockchain transactions with explainable AI insights.<div align="center">



## Tech Stack![XAI-Chain Logo](https://via.placeholder.com/150)



**Frontend:** Next.js 14, TypeScript, Tailwind CSS, RainbowKit, Wagmi  **Detect malicious blockchain transactions with AI • Explain every decision • Verify on-chain**

**Backend:** FastAPI, Python 3.11, XGBoost, SHAP  

**Blockchain:** Solidity, Hardhat, ethers.js  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Storage:** IPFS (Pinata)[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)](https://fastapi.tiangolo.com/)

## Features[![Solidity](https://img.shields.io/badge/Solidity-0.8.20-363636)](https://soliditylang.org/)



- Real-time transaction analysis using ML</div>

- Explainable AI with SHAP values

- Multi-chain wallet support## 🎯 Overview

- IPFS-based audit trail

- Smart contract verificationXAI-Chain is a production-ready Web3 + AI security platform that combines machine learning with explainable AI (XAI) to detect malicious blockchain transactions. Every prediction is explained using SHAP values and stored immutably on Polygon Mumbai testnet.

- Interactive dashboard

### Key Features

## Project Structure

- 🤖 **AI-Powered Detection**: XGBoost model trained on blockchain transaction patterns

```- 📊 **Explainable AI**: SHAP (SHapley Additive exPlanations) for transparency

Xplainable-blockchain/- ⛓️ **On-Chain Verification**: Immutable storage on Polygon Mumbai

├── backend/              # FastAPI server- 📦 **IPFS Storage**: Decentralized explanation data via Pinata

│   ├── app/- 🎨 **Modern UI**: Next.js 14 with Tailwind CSS and shadcn/ui

│   │   ├── main.py      # API entry- 🔗 **Web3 Integration**: RainbowKit + Wagmi for seamless wallet connection

│   │   ├── routers/     # Endpoints

│   │   ├── services/    # Blockchain and IPFS## 🏗️ Architecture

│   │   ├── utils/       # Feature engineering

│   │   └── ml/          # Trained models```

│   └── data/            # Training data┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐

├── frontend/            # Next.js app│   Frontend      │       │   Backend API    │       │   Blockchain    │

│   ├── app/│   (Next.js 14)  │◄─────►│   (FastAPI)      │◄─────►│   (Polygon)     │

│   │   ├── page.tsx           # Home│   + RainbowKit  │       │   + ML Models    │       │   + Smart       │

│   │   ├── analyze/page.tsx   # Analysis│                 │       │   + SHAP         │       │     Contract    │

│   │   └── dashboard/page.tsx # Dashboard└─────────────────┘       └──────────────────┘       └─────────────────┘

│   └── config/          # Web3 config                                   │

└── blockchain/          # Hardhat                                   ▼

    ├── contracts/       # Solidity                          ┌──────────────────┐

    ├── scripts/         # Deploy                          │   IPFS (Pinata)  │

    └── test/           # Tests                          │   Storage        │

```                          └──────────────────┘

```

## Quick Start

## 🚀 Quick Start

### Prerequisites

### Prerequisites

- Python 3.11+

- Node.js 16+- **Node.js** 20+ and npm

- WSL2 (for Windows)- **Python** 3.10+ and pip

- **MetaMask** wallet extension

### 1. Backend Setup- **API Keys** (see Setup section)



```powershell### Installation

cd backend

#### 1. Clone Repository

# create virtual environment

python -m venv venv```powershell

git clone https://github.com/yourusername/xai-chain.git

# activate itcd xai-chain

.\venv\Scripts\Activate.ps1```



# install packages#### 2. Blockchain Setup

pip install -r requirements.txt

```powershell

# generate training data# Navigate to blockchain directory

python data/generate_sample_data.pycd blockchain



# train ML model# Install dependencies

python train_model.pynpm install

```

# Copy environment file

### 2. Frontend Setupcopy .env.example .env



```powershell# Edit .env with your private key and Infura URL

cd frontend# Get Mumbai MATIC from: https://mumbaifaucet.com/



# install packages using WSL# Compile contracts

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/frontend && npm install"npx hardhat compile

```

# Deploy to Polygon Mumbai

### 3. Blockchain Setupnpx hardhat run scripts/deploy.js --network mumbai

```

```powershell

cd blockchain**Important**: Save the deployed contract address!



# install packages#### 3. Backend Setup

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npm install"

```powershell

# compile contracts# Navigate to backend directory

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npx hardhat compile"cd ..\backend



# run tests# Create virtual environment

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npx hardhat test"python -m venv venv

```

# Activate virtual environment

## Running the Application.\venv\Scripts\activate



### Start Backend# Install dependencies

pip install -r requirements.txt

```powershell

cd d:\CODING\Xplainable-blockchain\backend# Copy environment file

$env:PYTHONPATH="d:\CODING\Xplainable-blockchain\backend"copy .env.example .env

d:\CODING\Xplainable-blockchain\backend\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```# Edit .env with:

# - Contract address from step 2

Backend: **http://localhost:8000**# - Infura URL

# - Pinata API keys

### Start Frontend# - MongoDB URI

```

```powershell

cd d:\CODING\Xplainable-blockchain\frontend#### 4. Generate Sample Data & Train Model

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/frontend && npm run dev"

``````powershell

# Generate sample transaction data

Frontend: **http://localhost:3000**python data\generate_sample_data.py



### Start Local Blockchain (Optional)# This creates data/processed/features.csv with 10,000 sample transactions

```

```powershell

cd d:\CODING\Xplainable-blockchain\blockchain**Train the ML model:**

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npx hardhat node"

```Create a Jupyter notebook or Python script with:



## Configuration```python

import pandas as pd

### Backend (.env)import numpy as np

from sklearn.model_selection import train_test_split

```envfrom sklearn.preprocessing import StandardScaler

# blockchain settingsfrom sklearn.metrics import classification_report, roc_auc_score

INFURA_URL=https://polygon-mumbai.infura.io/v3/YOUR_KEYimport xgboost as xgb

PRIVATE_KEY=your_private_keyimport pickle

CONTRACT_ADDRESS=deployed_address

# Load data

# IPFS storagedf = pd.read_csv('data/processed/features.csv')

PINATA_JWT=your_jwt_token

# Prepare features

# database optionalfeature_columns = [

MONGODB_URI=mongodb://localhost:27017/xaichain    'gas_price', 'gas_used', 'value', 'gas_price_deviation',

    'sender_tx_count', 'contract_age', 'is_contract_creation',

# APIs    'function_signature_hash', 'block_gas_used_ratio'

ETHERSCAN_API_KEY=your_key]

```

X = df[feature_columns]

### Frontend (.env.local)y = df['is_malicious']



```env# Split data

NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID=your_idX_train, X_test, y_train, y_test = train_test_split(

NEXT_PUBLIC_INFURA_API_KEY=your_key    X, y, test_size=0.2, random_state=42, stratify=y

NEXT_PUBLIC_API_URL=http://localhost:8000)

```

# Scale features

## Deploy Smart Contractscaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

### Local DeploymentX_test_scaled = scaler.transform(X_test)



Terminal 1 - Start node:# Train XGBoost

```powershellmodel = xgb.XGBClassifier(

cd blockchain    n_estimators=100,

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npx hardhat node"    max_depth=6,

```    learning_rate=0.1,

    random_state=42

Terminal 2 - Deploy:)

```powershellmodel.fit(X_train_scaled, y_train)

cd blockchain

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npx hardhat run scripts/deploy.js --network localhost"# Evaluate

```y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

Copy the address and update `backend/.env`:

# Save model and scaler

```envwith open('app/ml/model.pkl', 'wb') as f:

INFURA_URL=http://127.0.0.1:8545    pickle.dump(model, f)

CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3with open('app/ml/scaler.pkl', 'wb') as f:

PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80    pickle.dump(scaler, f)

```

print("✅ Model saved!")

Restart backend to use real blockchain.```



## Using the Platform#### 5. Frontend Setup



1. Open http://localhost:3000```powershell

2. Connect wallet (MetaMask, Rainbow, etc)# Navigate to frontend directory

3. Go to "Analyze Transaction"cd ..\frontend

4. Enter transaction hash

5. Click "Analyze Transaction"# Install dependencies

6. View risk analysisnpm install



## API Endpoints# Copy environment file

copy .env.local.example .env.local

**Health Check**

```# Edit .env.local with:

GET http://localhost:8000/# - API URL (http://localhost:8000)

```# - Contract address

# - WalletConnect Project ID (get from https://cloud.walletconnect.com)

**Analyze Transaction**```

```

POST http://localhost:8000/api/analyze/### Running the Application

```

Open **3 terminal windows**:

Request:

```json**Terminal 1 - Backend:**

{```powershell

  "tx_hash": "0xbf7ba7b25e4fbca4ee0ff460dba1c9ccaf5195e9c2ea24909b0f6e605636ceac",cd backend

  "network": "ethereum",.\venv\Scripts\activate

  "transaction_data": {uvicorn app.main:app --reload

    "amount": 0.01040272,```

    "gas_price": 0.117164461,API will run at: http://localhost:8000

    "gas_used": 21000,

    "num_transfers": 1,**Terminal 2 - Frontend:**

    "unique_addresses": 2,```powershell

    "time_of_day": 10,cd frontend

    "contract_interaction": 0,npm run dev

    "sender_tx_count": 150,```

    "receiver_tx_count": 75Frontend will run at: http://localhost:3000

  }

}**Terminal 3 - Optional monitoring:**

``````powershell

# View logs, check contracts, etc.

**Get Audit Trail**```

```

GET http://localhost:8000/api/audit/### Access the Application

```

1. Open browser to http://localhost:3000

**Verify Transaction**2. Connect your MetaMask wallet (switch to Polygon Mumbai)

```3. Click "Analyze Transaction"

GET http://localhost:8000/api/verify/{tx_hash}4. Enter a transaction hash

```5. View AI prediction + SHAP explanation + on-chain verification!



## ML Model Details## 📖 API Documentation



### Features (9 total)Once the backend is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs

1. amount - Transaction value in ETH- **ReDoc**: http://localhost:8000/redoc

2. gas_price - Gas price in Gwei

3. gas_used - Gas consumed### Main Endpoints

4. num_transfers - Token transfer count

5. unique_addresses - Unique address count#### Analyze Transaction

6. time_of_day - Hour (0-23)```http

7. contract_interaction - Boolean flagPOST /api/analyze

8. sender_tx_count - Sender historyContent-Type: application/json

9. receiver_tx_count - Receiver history

{

### Performance  "tx_hash": "0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060",

  "network": "ethereum"

- Algorithm: XGBoost}

- Training samples: 10,000```

- Test accuracy: 100%

- Explainability: SHAP values#### Verify Explanation

```http

### Retrain ModelGET /api/verify/{tx_hash}

```

```powershell

cd backend#### Get Audit Trail

.\venv\Scripts\Activate.ps1```http

python train_model.pyGET /api/audit?limit=10&skip=0

``````



## Troubleshooting## 🧪 Testing



### Frontend Chunk Error### Smart Contract Tests



```powershell```powershell

cd frontendcd blockchain

Remove-Item -Recurse -Force .nextnpx hardhat test

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/frontend && npm run dev"```

```

### Backend Tests

### Backend Module Not Found

```powershell

```powershellcd backend

cd backendpytest tests/

$env:PYTHONPATH="d:\CODING\Xplainable-blockchain\backend"```

d:\CODING\Xplainable-blockchain\backend\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```## 📦 Project Structure



### Port Already in Use```

xai-chain/

Stop processes:├── frontend/                  # Next.js 14 frontend

```powershell│   ├── app/                  # App router pages

# stop backend│   ├── components/           # React components

Get-Process python | Where-Object {$_.CommandLine -like "*uvicorn*"} | Stop-Process -Force│   ├── config/               # Wagmi configuration

│   └── lib/                  # Utilities

# stop frontend│

Get-Process | Where-Object {$_.ProcessName -eq "node"} | Stop-Process -Force├── backend/                  # FastAPI backend

```│   ├── app/

│   │   ├── main.py          # Entry point

### WSL Issues│   │   ├── routers/         # API endpoints

│   │   ├── models/          # AI models & schemas

```powershell│   │   ├── services/        # Blockchain & IPFS

wsl --shutdown│   │   └── utils/           # Helpers

```│   └── requirements.txt

│

Wait 10 seconds and retry.├── blockchain/               # Smart contracts

│   ├── contracts/           # Solidity files

## Contract Tests│   ├── scripts/             # Deployment scripts

│   ├── test/                # Contract tests

```powershell│   └── hardhat.config.js

cd blockchain│

wsl bash -c "cd /mnt/d/CODING/Xplainable-blockchain/blockchain && npx hardhat test"└── data/                    # Data & models

```    ├── generate_sample_data.py

    └── processed/           # Training data

All 6 tests should pass.```



## Architecture## 🔑 Environment Variables



### Data Flow### Blockchain (.env)

```env

1. User submits transaction hashPOLYGON_MUMBAI_RPC=https://polygon-mumbai.infura.io/v3/YOUR_KEY

2. Frontend sends request to backendPRIVATE_KEY=your_private_key_without_0x

3. Backend extracts 9 featuresPOLYGONSCAN_API_KEY=your_polygonscan_key

4. XGBoost model predicts risk```

5. SHAP generates explanation

6. Result stored on IPFS### Backend (.env)

7. Result stored on blockchain (optional)```env

8. Frontend displays analysisINFURA_URL=https://polygon-mumbai.infura.io/v3/YOUR_KEY

PRIVATE_KEY=your_private_key_without_0x

### SecurityCONTRACT_ADDRESS=0x_deployed_contract_address

PINATA_API_KEY=your_pinata_key

- Private keys not in frontendPINATA_API_SECRET=your_pinata_secret

- API rate limitingMONGODB_URI=mongodb://localhost:27017/xaichain

- Smart contracts testedETHERSCAN_API_KEY=your_etherscan_key

- IPFS content addressed```

- Blockchain immutability

### Frontend (.env.local)

## Current Status```env

NEXT_PUBLIC_API_URL=http://localhost:8000

- ML Model: TrainedNEXT_PUBLIC_CONTRACT_ADDRESS=0x_deployed_contract_address

- Backend API: WorkingNEXT_PUBLIC_WALLETCONNECT_PROJECT_ID=your_walletconnect_id

- Frontend UI: 3 pages complete```

- Smart Contracts: Compiled and tested

- Blockchain: Configurable (mocked by default)## 🎓 How It Works

- IPFS: Configurable (mocked by default)

1. **User submits transaction hash** via frontend

## Example Transaction2. **Backend fetches transaction data** from blockchain

3. **Features are extracted** (gas price, value, sender history, etc.)

Try this real Ethereum transaction:4. **XGBoost model predicts** if transaction is malicious

```5. **SHAP generates explanation** showing feature importance

0xbf7ba7b25e4fbca4ee0ff460dba1c9ccaf5195e9c2ea24909b0f6e605636ceac6. **Explanation uploaded to IPFS** via Pinata

```7. **Smart contract stores** IPFS hash + risk score on-chain

8. **Frontend displays** prediction, explanation, and verification links

Details:

- From: buildernet.eth## 🛠️ Tech Stack

- To: 0x3cEFB98AbB3ebf3AFAFF3A377285fe6d4eA3907e

- Value: 0.01040272 ETH| Layer | Technology |

- Gas Price: 0.117164461 Gwei|-------|-----------|

- Block: 23696643| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui |

| Web3 | ethers.js v6, Wagmi v2, RainbowKit |

## License| Backend | FastAPI, Uvicorn, Python 3.10+ |

| AI/ML | XGBoost, SHAP, scikit-learn, pandas, numpy |

MIT| Blockchain | Solidity 0.8.20, Hardhat, Polygon Mumbai |

| Storage | IPFS (Pinata), MongoDB |

---| APIs | Web3.py, Infura, Etherscan |



Built for blockchain security with explainable AI.## 🤝 Contributing


Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SHAP**: For explainable AI framework
- **Hardhat**: For Ethereum development environment
- **RainbowKit**: For beautiful wallet connection UI
- **shadcn/ui**: For beautiful UI components
- **Pinata**: For IPFS pinning service

## 📞 Support

- **Documentation**: [https://xai-chain-docs.vercel.app](https://xai-chain-docs.vercel.app)
- **Issues**: [GitHub Issues](https://github.com/yourusername/xai-chain/issues)
- **Discord**: [Join our community](#)
- **Email**: support@xai-chain.io

## 🗺️ Roadmap

- [ ] Support for multiple blockchains (Ethereum, BSC, etc.)
- [ ] Real-time monitoring dashboard
- [ ] Batch transaction analysis
- [ ] Advanced XAI visualizations (LIME, counterfactuals)
- [ ] Mobile app (React Native)
- [ ] Historical analytics and trends
- [ ] API rate limiting and authentication
- [ ] Mainnet deployment

---

<div align="center">

**Built with ❤️ for the Web3 community**

[Website](#) • [Documentation](#) • [Twitter](#) • [Discord](#)

</div>
