# TeeTee - Decentralized AI Inference with TEE & 0G Network

<div align="center">

[![Live App](https://img.shields.io/badge/Live_App-teetee.site-blue)](https://teetee.site)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![0G Network](https://img.shields.io/badge/Built_on-0G_Network-purple)](https://0g.ai)
[![WaveHack](https://img.shields.io/badge/0G_WaveHack-Wave_5_Winner-gold)](https://0g.ai)

**Democratizing AI through secure, verifiable, and decentralized inference powered by Trusted Execution Environments**

[Live Demo](https://teetee.site) • [Documentation](https://docs.google.com/document/d/1pqDrJoYoBfVG19Kxu0-9uSHfwEq3ZQjp8d1CU9Pd-Kk/edit?usp=sharing) • [Video Demo](https://drive.google.com/drive/folders/1eWDgBJ_o2jr5xT2U_ZYclxhAAt15G4HJ?usp=sharing) • [Twitter Thread](https://x.com/ilovedahmo/status/1986064335354126573)

🏆 **0G WaveHack Wave 5 Submission** | Production-Ready | Mainnet Deployed | Fully Verified

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Verified Mainnet Contracts](#-verified-mainnet-contracts-0g-network)
- [Unique Selling Point (USP)](#-unique-selling-point-usp)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Frontend Setup](#1-frontend-setup)
  - [Backend Setup](#2-backend-setup)
  - [Smart Contract Deployment (Optional)](#3-smart-contract-deployment-optional)
  - [iNFT Deployment (Optional)](#4-inft-deployment-optional)
- [Project Structure](#-project-structure)
- [Environment Variables](#-environment-variables)
- [Development](#-development)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [Community & Support](#-community--support)
- [License](#-license)

---

## 🌟 Overview

**TeeTee** is a production-ready decentralized AI inference platform that brings together blockchain technology, Trusted Execution Environments (TEE), and the 0G Network to create a secure, transparent, and accessible AI ecosystem. Our platform enables users to host, access, and monetize Large Language Models (LLMs) while maintaining privacy and verifiability.

### Key Innovations

- **🔐 TEE-Secured Inference**: All AI computations run in hardware-isolated Trusted Execution Environments, ensuring data privacy and tamper-proof execution
- **⚡ 0G Network Integration**: Leverages 0G's high-performance storage (2GB/s) and mainnet smart contracts for on-chain authorization and credit management
- **💎 iNFT Token Standard**: Implements ERC-7857 for tokenized AI agent ownership with secure, privacy-preserving metadata transfer
- **🏥 Auto-Failover System**: Health oracle continuously monitors shard status and automatically routes traffic to healthy endpoints
- **💰 Credit-Based Economy**: Fair, transparent metering and settlement system for AI usage with verifiable on-chain transactions

### 📜 Verified Mainnet Contracts (0G Network)

All core contracts are deployed and verified on 0G mainnet:

| Contract | Address | Purpose | Explorer Link |
|----------|---------|---------|---------------|
| **CreditUse** | `0xd1ce92b2c95a892fe1166e20b65c73b33b269f7e` | Credit management and usage metering | [View Contract](https://chainscan.0g.ai/address/0xd1ce92b2c95a892fe1166e20b65c73b33b269f7e?tab=contract-viewer) |
| **OracleStub** | `0x20f8f585f8e0d3d1fce7907a3c02aeaa5c924707` | Oracle interface for health checking | [View Contract](https://chainscan.0g.ai/address/0x20f8f585f8e0d3d1fce7907a3c02aeaa5c924707?tab=contract-viewer) |
| **DataVerifierAdapterFixed** | `0x8889106de495dc1731a9b60a58817de6e0142ac0` | Data verification adapter | [View Contract](https://chainscan.0g.ai/address/0x8889106de495dc1731a9b60a58817de6e0142ac0?tab=contract-viewer) |
| **INFT (ERC-7857)** | `0x56776a7878c7d4cc9943b17d91a3e098c77614da` | Intelligent NFT for AI agents | [View Contract](https://chainscan.0g.ai/address/0x56776a7878c7d4cc9943b17d91a3e098c77614da?tab=contract-viewer) |

🔗 **Building Journey & Mainnet Deployment**: [Read our Twitter thread](https://x.com/ilovedahmo/status/1986064335354126573) documenting key milestones and deployment details.

---

## 🎯 Unique Selling Point (USP)

### What Makes TeeTee Stand Out in the 0G Ecosystem?

**TeeTee is the first and only production-ready AI inference platform that combines hardware-level security (TEE), fully on-chain credit settlement, and intelligent failover—all optimized for 0G's high-performance infrastructure.**

#### 1. **True Privacy-Preserving AI with TEE + 0G Storage**
Unlike other AI platforms that claim decentralization but rely on centralized inference servers, TeeTee runs **100% of inference inside Trusted Execution Environments (TEE)**. This hardware-isolated compute ensures:
- 🔒 **Zero Data Leakage**: Your prompts and model outputs never touch unencrypted memory
- 🛡️ **Tamper-Proof Execution**: Even we (the platform operators) cannot access user data
- ✅ **Cryptographic Verification**: Every inference can be proven to have executed in a genuine TEE

Combined with **0G Storage's decentralized architecture**, we store model weights and metadata in a trustless, verifiable manner—making TeeTee the most secure AI platform in Web3.

#### 2. **First ERC-7857 (iNFT) Implementation on 0G**
TeeTee pioneers **Intelligent NFTs** on 0G Network, enabling:
- 🎨 **AI Agent Ownership**: Tokenize and trade AI models as NFTs with embedded intelligence
- 🔐 **Secure Metadata Transfer**: When an iNFT is sold, the AI model, memory, and traits transfer securely via oracle-assisted encryption
- 💼 **New Economic Models**: Create AI agent marketplaces, rental systems, and royalty structures impossible with traditional NFTs

This positions TeeTee as the **infrastructure layer for the future AI agent economy** on 0G.

#### 3. **Production-Grade Reliability with Auto-Failover**
Most decentralized AI projects struggle with reliability. TeeTee solves this with:
- 🏥 **Health Oracle**: Continuously monitors all TEE endpoints (every 30 seconds)
- 🔄 **Automatic Failover**: Detects and routes around unhealthy shards in real-time
- 📊 **Transparent SLAs**: On-chain proof of uptime and performance metrics
- ⚡ **Sub-500ms Latency**: Optimized for 0G Chain's 2,500+ TPS and sub-second finality

We're not just an experiment—**TeeTee is live, stable, and serving real traffic at [teetee.site](https://teetee.site)**.

#### 4. **Developer-First with OpenAI-Compatible API**
TeeTee is **drop-in compatible** with OpenAI's API, meaning:
- 🚀 **Zero Migration Cost**: Change one line of code to switch from OpenAI to TeeTee
- 📚 **Familiar Interface**: Use existing tools, libraries, and workflows
- 🌐 **Web3-Native**: Built-in wallet support, on-chain credit management, and gasless transactions

This makes TeeTee the **easiest onboarding path** for Web2 developers entering the 0G ecosystem.

#### 5. **Fair, Transparent Credit Economy**
Unlike opaque subscription models, TeeTee uses:
- 💰 **Pay-Per-Token**: Only pay for what you use, down to the token level
- 📊 **On-Chain Verification**: Every credit transaction is verifiable on 0G mainnet
- 🎁 **Hoster Incentives**: Model hosters earn directly from usage with automated settlements
- 🔓 **No Lock-In**: Credits are portable and can be withdrawn anytime

Our credit system aligns incentives between users, hosters, and the platform—**creating a sustainable, decentralized AI economy**.

#### 6. **Battle-Tested with Real-World Usage**
TeeTee completed **0G WaveHack Wave 5** with:
- ✅ **Mainnet Deployment**: All core contracts live on 0G mainnet
- ✅ **Production Dapp**: Fully functional UI at [teetee.site](https://teetee.site)
- ✅ **Full Test Coverage**: Unit + integration tests across contracts, oracle, and frontend
- ✅ **Verified Contracts**: All addresses published and verified on 0G Chainscan
- ✅ **Public Documentation**: Hosting guides, FAQs, and API references

We're not vaporware—**TeeTee is live, audited, and ready to scale**.

---

## 🏗 Architecture

TeeTee is built on a modular, full-stack architecture consisting of five core components:

### 1. **Frontend** (`/frontend`)
**Purpose**: User-facing Next.js application providing the interface for AI inference and LLM management

**Key Features**:
- Modern, responsive UI built with React 19, Next.js 16, and Tailwind CSS
- Spline-powered 3D visualizations for enhanced UX
- Comprehensive Console for adding/managing LLMs
- Hoster Dashboard with earnings analytics and shard status monitoring
- Model marketplace with filters, per-model icons, and usage insights
- Secure wallet integration via RainbowKit and wagmi
- Real-time credit management and withdrawal flows

**Tech Stack**: Next.js, React, TypeScript, Tailwind CSS, ethers.js, RainbowKit, Spline

---

### 2. **Backend** (`/backend`)
**Purpose**: Oracle service that bridges off-chain TEE computation with on-chain smart contracts

**Key Features**:
- Health checking service for TEE endpoints
- Shard lifecycle management (registration, validation, failover)
- Model hash verification for authenticity
- RESTful API for inference routing
- Duplicate URL prevention and host validation
- Integration with 0G Network for decentralized storage
- Circuit breaker pattern for resilience

**Tech Stack**: Node.js, TypeScript, Express, ethers.js, 0G-TS-SDK

---

### 3. **Smart Contracts** (`/smartcontract`)
**Purpose**: On-chain logic for credit management and usage metering

**Key Components**:
- **`creditUse.sol`**: Manages LLM credit consumption, authorization, and metering. Implements pay-per-token model with verifiable on-chain settlement
- **`subscription.sol`**: Handles subscription plans and recurring payment logic

**Blockchain**: Deployed on 0G mainnet (EVM-compatible Layer 1)

**Key Features**:
- Gas-optimized operations (storage packing, custom errors, selective events)
- Safe math operations with overflow protection
- Credit minting, transfer, and burn mechanisms
- Per-shard usage tracking and settlement

---

### 4. **0G iNFT** (`/0g-INFT`)
**Purpose**: Implementation of ERC-7857 standard for tokenizing AI agents with secure metadata transfer

**Key Features**:
- Privacy-preserving metadata encryption
- Secure transfer protocol using trusted oracles and TEEs
- Integration with 0G Storage for decentralized model hosting
- Ownership-based authorization for hosted LLMs
- Dynamic metadata management for evolving AI agents
- Hash commitment and cryptographic proof verification

**Smart Contracts**:
- **`INFT.sol`**: Core ERC-7857 implementation
- **`INFTFixed.sol`**: Optimized version with latest enhancements
- **`OracleStub.sol`**: Oracle interface for metadata verification

**Use Cases**:
- Tokenized AI agent ownership and trading
- Access control for premium models
- Verifiable provenance of AI models
- Decentralized AI marketplaces

---

### 5. **Model Splitting Research** (`/ModelSplitting-v1`)
**Purpose**: Experimental research on distributed model inference across multiple shards

**Key Concepts**:
- Algorithm for splitting large language models into distributable shards
- Load balancing strategies for parallel inference
- Coordination protocols for multi-shard computation
- Proof-of-concept implementation in Python

**Research Goals**:
- Reduce single-point-of-failure risks
- Enable horizontal scaling of massive models
- Improve inference latency through parallelization
- Explore federated learning possibilities

**Status**: Early-stage research, not yet integrated into production

---

### 6. **LLM Server** (`/llm`)
**Purpose**: Node.js server for running LLM inference with OpenAI-compatible API interface

**Features**:
- Phala Cloud GPU TEE integration for confidential AI
- Support for multiple models (DeepSeek V3, Llama 3.3, GPT-OSS, Qwen)
- Streaming and non-streaming response modes
- RESTful endpoints (`/chat`, `/chat/stream`, `/inference`, `/models`)
- Hardware-level privacy protection with cryptographic execution proofs

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Console    │  │   Models     │  │   Hoster Dashboard   │  │
│  │  (Add LLMs)  │  │ (Marketplace)│  │  (Earnings/Status)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend Oracle Service                        │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Health Checker │  │  Shard Registry │  │ Inference Router│  │
│  │   (Failover)   │  │   (Validation)  │  │  (Load Balance) │  │
│  └────────────────┘  └─────────────────┘  └─────────────────┘  │
└───────────┬─────────────────────┬──────────────────┬────────────┘
            │                     │                  │
            ▼                     ▼                  ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Smart Contracts │   │    0G Storage    │   │   TEE Hosts      │
│   (0G Mainnet)   │   │  (Decentralized) │   │ (Phala Cloud)    │
│                  │   │                  │   │                  │
│ • creditUse.sol  │   │ • Model Data     │   │ • DeepSeek V3    │
│ • subscription   │   │ • Metadata       │   │ • Llama 3.3      │
│ • INFT (ERC-7857)│   │ • Health Logs    │   │ • Qwen, GPT-OSS  │
└──────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## ✨ Features

### For AI Consumers
- 🤖 **Access Premium Models**: Query state-of-the-art LLMs (DeepSeek V3, Llama 3.3, etc.) through a simple interface
- 💳 **Credit-Based Payments**: Fair, transparent pay-per-token pricing with on-chain verification
- 🔒 **Privacy Guaranteed**: All inference runs in TEE with hardware-level isolation
- ⚡ **Low Latency**: Auto-failover ensures queries route to healthy, fast endpoints
- 📊 **Usage Analytics**: Track your credit consumption and query history

### For Model Hosters
- 💰 **Monetize Models**: Earn rewards by hosting LLMs and serving inference requests
- 📈 **Earnings Dashboard**: Real-time analytics on usage, revenue, and shard performance
- 🛡️ **Automated Health Checks**: Oracle monitors your endpoints and handles failover
- 🎯 **Easy Onboarding**: Register shards through intuitive Console interface
- 🔐 **Ownership Control**: iNFT-based authorization ensures only you control your models

### For Developers
- 🔧 **OpenAI-Compatible API**: Drop-in replacement for existing AI applications
- 📚 **Comprehensive SDKs**: TypeScript/JavaScript support with 0G-TS-SDK integration
- 🌐 **Cross-Chain Ready**: EVM-compatible, works with existing Web3 infrastructure
- 🧪 **Full Test Coverage**: Unit and integration tests across contracts, oracle, and frontend
- 📖 **Detailed Documentation**: Hosting guides, API references, and troubleshooting FAQs

---

## 🛠 Tech Stack

### Frontend
- **Framework**: Next.js 16 (Turbopack), React 19
- **Styling**: Tailwind CSS 4, Framer Motion, Spline
- **Web3**: ethers.js 6, RainbowKit, wagmi, viem
- **Storage**: 0G-TS-SDK for decentralized storage
- **UI Libraries**: Radix UI, Lucide Icons, GSAP

### Backend
- **Runtime**: Node.js with TypeScript
- **Framework**: Express.js
- **Web3**: ethers.js 6, 0G-TS-SDK
- **Reliability**: Opossum (circuit breaker), express-rate-limit
- **Communication**: Server-Sent Events (SSE) for streaming

### Smart Contracts
- **Language**: Solidity ^0.8.0
- **Development**: Hardhat
- **Network**: 0G Mainnet (EVM-compatible)
- **Standards**: ERC-7857 (iNFT), ERC-20 (credits)

### Infrastructure
- **Blockchain**: 0G Network (2,500+ TPS, sub-second finality)
- **Storage**: 0G Storage (2GB/s throughput, ~$10/TB)
- **Compute**: Phala Cloud GPU TEE
- **Verification**: Proof-of-Inference (PoI), Proof-of-Random-Access (PoRA)

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** v20+ and npm (or yarn)
- **Git** for cloning the repository
- **MetaMask** or compatible Web3 wallet
- **0G Network RPC** access (for mainnet interaction)

---

### 1. Frontend Setup

The frontend is a Next.js application that provides the user interface for TeeTee.

#### Step 1: Navigate to Frontend Directory
```bash
cd frontend
```

#### Step 2: Install Dependencies
```bash
npm install
```

#### Step 3: Configure Environment Variables
Create a `.env.local` file in the `frontend` directory:

```env
# No environment variables required for basic frontend operation
# The frontend connects to public 0G RPC endpoints by default

# Optional: If using custom backend URL
NEXT_PUBLIC_BACKEND_URL=http://localhost:3002

# Optional: If using custom smart contract addresses
NEXT_PUBLIC_CREDIT_CONTRACT=0x...
NEXT_PUBLIC_INFT_CONTRACT=0x...
```

#### Step 4: Run Development Server
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

#### Step 5: Build for Production
```bash
npm run build
npm start
```

---

### 2. Backend Setup

The backend oracle service handles health checking, shard management, and inference routing.

#### Step 1: Navigate to Backend Directory
```bash
cd backend
```

#### Step 2: Install Dependencies
```bash
npm install
```

#### Step 3: Configure Environment Variables
Create a `.env` file in the `backend` directory:

```env
# Private key for signing transactions (REQUIRED)
PRIVATE_KEY=your_wallet_private_key_here

# RedPill API key for TEE verification (REQUIRED)
REDPILL_API_KEY=your_redpill_api_key_here

# Optional: Custom RPC endpoint
RPC_URL=https://evmrpc-testnet.0g.ai

# Optional: Backend port
PORT=3002

# Optional: Smart contract addresses (if not using defaults)
CREDIT_CONTRACT=0x...
INFT_CONTRACT=0x...
REGISTRY_CONTRACT=0x...
```

**⚠️ Security Warning**: Never commit your `.env` file or share your private key!

#### Step 4: Run Development Server
```bash
npm run dev
```

#### Step 5: Build and Run Production Server
```bash
npm run build
npm run serve
```

The backend API will be available at `http://localhost:3002`

---

### 3. Smart Contract Deployment (Optional)

Only required if you want to deploy your own instances of the credit and subscription contracts.

#### Step 1: Navigate to Smart Contract Directory
```bash
cd smartcontract
```

#### Step 2: Install Dependencies
```bash
npm install
```

#### Step 3: Configure Environment Variables
Create a `.env` file in the `smartcontract` directory:

```env
# Private key for deploying contracts
PRIVATE_KEY=your_wallet_private_key_here

# 0G Network RPC URL
RPC_URL=https://evmrpc-testnet.0g.ai

# Optional: Etherscan API key for verification
ETHERSCAN_API_KEY=your_etherscan_api_key_here
```

#### Step 4: Deploy Contracts
```bash
# Deploy to 0G testnet
npx hardhat run scripts/deploy.js --network galileo

# Deploy to 0G mainnet
npx hardhat run scripts/deploy.js --network mainnet
```

#### Step 5: Verify Contracts (Optional)
```bash
npx hardhat verify --network mainnet DEPLOYED_CONTRACT_ADDRESS "constructor_args"
```

---

### 4. iNFT Deployment (Optional)

Only required if you want to deploy your own iNFT (ERC-7857) contracts for AI agent tokenization.

#### Step 1: Navigate to 0G-INFT Directory
```bash
cd 0g-INFT
```

#### Step 2: Install Dependencies
```bash
npm install
```

#### Step 3: Configure Environment Variables
Create a `.env` file in the `0g-INFT` directory:

```env
# Private key for deploying contracts
PRIVATE_KEY=your_wallet_private_key_here

# 0G Network RPC URL
RPC_URL=https://evmrpc-testnet.0g.ai

# Optional: 0G Storage configuration
STORAGE_RPC_URL=https://storage-rpc.0g.ai
STORAGE_FLOW_ADDRESS=0x...

# Optional: Oracle configuration
ORACLE_ADDRESS=0x...
```

#### Step 4: Deploy iNFT Contracts
```bash
# Using deployment script
./deploy.sh

# Or manually with Hardhat
npx hardhat run scripts/deploy-inft.ts --network galileo
```

#### Step 5: Verify Deployment
```bash
npx hardhat run scripts/verify-deployment.ts --network galileo
```

**📖 For detailed deployment instructions**, see [`0g-INFT/DEPLOYMENT-GUIDE.md`](0g-INFT/DEPLOYMENT-GUIDE.md)

---

## 📁 Project Structure

```
TeeTee-v2/
├── frontend/                  # Next.js user interface
│   ├── components/            # React components (Console, Models, Dashboard)
│   ├── pages/                 # Next.js pages and API routes
│   ├── lib/                   # Contract ABIs and Web3 utilities
│   ├── utils/                 # Helper functions and hooks
│   └── styles/                # Global CSS and Tailwind config
│
├── backend/                   # Oracle service (Node.js/Express)
│   ├── index.ts               # Main server entry point
│   ├── networkConfig.ts       # 0G Network configuration
│   └── dist/                  # Compiled JavaScript output
│
├── smartcontract/             # Credit & subscription contracts
│   ├── contracts/             # Solidity smart contracts
│   │   ├── creditUse.sol      # Credit management
│   │   └── subscription.sol   # Subscription logic
│   ├── scripts/               # Deployment and utility scripts
│   └── hardhat.config.js      # Hardhat configuration
│
├── 0g-INFT/                   # ERC-7857 iNFT implementation
│   ├── contracts/             # Solidity smart contracts
│   │   ├── INFT.sol           # Core ERC-7857 implementation
│   │   ├── INFTFixed.sol      # Optimized version
│   │   └── OracleStub.sol     # Oracle interface
│   ├── scripts/               # Deployment and testing scripts
│   ├── deployments/           # Deployment artifacts
│   ├── storage/               # Encrypted metadata storage utilities
│   ├── frontend/              # iNFT demo frontend
│   └── offchain-service/      # Oracle service for metadata transfer
│
├── ModelSplitting-v1/         # Research on distributed inference
│   ├── main.py                # Main execution script
│   └── split.py               # Model splitting algorithm
│
├── llm/                       # LLM inference server
│   ├── server.js              # OpenAI-compatible API server
│   ├── Dockerfile             # Docker containerization
│   └── docker-compose.yml     # Multi-service orchestration
│
└── README.md                  # This file
```

---

## 🔐 Environment Variables

### Backend Environment Variables (Required for Basic Operation)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `PRIVATE_KEY` | Private key for signing transactions | ✅ Yes | `0xabc123...` |
| `REDPILL_API_KEY` | API key for RedPill TEE verification | ✅ Yes | `rp_abc123...` |
| `RPC_URL` | 0G Network RPC endpoint | ❌ No | `https://evmrpc-testnet.0g.ai` |
| `PORT` | Backend server port | ❌ No | `3002` |

### Smart Contract Deployment Variables (Optional)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `PRIVATE_KEY` | Private key for deploying contracts | ✅ Yes | `0xabc123...` |
| `RPC_URL` | 0G Network RPC endpoint | ✅ Yes | `https://evmrpc-testnet.0g.ai` |
| `ETHERSCAN_API_KEY` | For contract verification | ❌ No | `ABC123...` |

### iNFT Deployment Variables (Optional)

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `PRIVATE_KEY` | Private key for deploying iNFT contracts | ✅ Yes | `0xabc123...` |
| `RPC_URL` | 0G Network RPC endpoint | ✅ Yes | `https://evmrpc-testnet.0g.ai` |
| `STORAGE_RPC_URL` | 0G Storage RPC endpoint | ❌ No | `https://storage-rpc.0g.ai` |
| `ORACLE_ADDRESS` | Trusted oracle contract address | ❌ No | `0x...` |

---

## 💻 Development

### Running Tests

#### Frontend Tests
```bash
cd frontend
npm test
```

#### Backend Tests
```bash
cd backend
npm test
```

#### Smart Contract Tests
```bash
cd smartcontract
npx hardhat test
```

#### iNFT Contract Tests
```bash
cd 0g-INFT
npx hardhat test
```

### Linting and Formatting

```bash
# Frontend
cd frontend
npm run lint

# Backend
cd backend
npm run lint
```

### Building All Components

```bash
# Build frontend
cd frontend && npm run build

# Build backend
cd backend && npm run build

# Compile smart contracts
cd smartcontract && npx hardhat compile

# Compile iNFT contracts
cd 0g-INFT && npx hardhat compile
```

---

## 🗺 Future Roadmap

TeeTee has successfully completed the 0G WaveHack Wave 5, deploying mainnet contracts, launching a production dapp at [teetee.site](https://teetee.site), and hardening the oracle infrastructure. Our future roadmap builds on this foundation to scale toward GA and beyond.

### Phase 1: Public Beta → General Availability (Q1-Q2 2025)

#### Production Hardening
- **SLA Implementation**: Establish and enforce service-level agreements
  - 99.9% uptime guarantee for inference endpoints
  - < 500ms p95 latency for standard queries
  - Automated compensation for SLA breaches
- **Autoscaling Policies**: Dynamic resource allocation based on load
  - Horizontal scaling of TEE compute nodes
  - Intelligent shard activation/deactivation
  - Predictive load balancing using historical patterns
- **Advanced Monitoring**: Comprehensive observability stack
  - Real-time dashboards for shard health, latency, and throughput
  - Alerting system for anomaly detection
  - Performance analytics for hosters and users

#### Developer Experience
- **SDKs & Client Libraries**
  - JavaScript/TypeScript SDK with full type safety
  - Python SDK for ML/AI workflows
  - Go SDK for high-performance applications
  - REST API with OpenAPI/Swagger documentation
- **Quickstart Templates**: Boilerplate projects for common use cases
  - AI chatbot starter (React, Vue, Svelte)
  - DeFi AI agent template
  - Gaming NPC AI integration
- **Interactive Tutorials**: Hands-on guides for developers
  - "Build Your First AI-Powered dApp in 15 Minutes"
  - Advanced patterns (RAG, fine-tuning, multi-model orchestration)
- **Partner Onboarding Program**: Collaborative integration with Web3 protocols
  - Co-marketing and technical support
  - Custom integration assistance
  - Grant program for innovative use cases

#### Economic Model Enhancements
- **Tiered Subscription Plans**
  - Free tier: 100 credits/month for experimentation
  - Pro tier: Discounted bulk credits + priority routing
  - Enterprise tier: Custom SLAs, dedicated support, volume discounts
- **Stake-Based Incentives**: Reward long-term participants
  - Hosters stake tokens to boost earnings and reputation
  - Users stake for reduced inference fees
- **Dynamic Pricing**: Market-driven credit costs
  - Surge pricing during high demand
  - Off-peak discounts to optimize utilization
- **Referral Program**: Growth incentives for community advocates

---

### Phase 2: Model Ecosystem Expansion (Q2-Q3 2025)

#### Expanded Model Support
- **Multi-Modal Models**
  - Image generation (Stable Diffusion, DALL-E alternatives)
  - Image understanding (CLIP, LLaVA)
  - Audio generation and transcription (Whisper, Bark)
  - Video analysis and generation
- **Specialized Domain Models**
  - Code generation (CodeLlama, StarCoder, WizardCoder)
  - Scientific reasoning (Galactica, BioGPT)
  - Legal and financial analysis
  - Medical diagnosis assistants (with compliance safeguards)
- **Fine-Tuned Custom Models**: Marketplace for user-trained models
  - Upload and monetize custom fine-tunes
  - Domain-specific adapters (LoRA, QLoRA)
  - Private model hosting with access control

#### Model Splitting & Distributed Inference (Research → Production)
- **Productionize Research**: Transition ModelSplitting-v1 to live system
  - Algorithm refinement for production workloads
  - Fault-tolerant shard coordination
  - Latency optimization for multi-shard inference
- **Federated Learning Support**: Collaborative model training
  - Privacy-preserving gradient aggregation
  - Decentralized training across hosters
  - On-chain proof of training contribution
- **Elastic Scaling**: Auto-split large models across multiple shards
  - Support for 100B+ parameter models (LLaMA 2 70B, GPT-J 6B, etc.)
  - Dynamic shard allocation based on query complexity

---

### Phase 3: Enhanced Decentralization & Security (Q3-Q4 2025)

#### Trustless Oracle Network
- **Multi-Oracle Consensus**: Eliminate single points of failure
  - Decentralized oracle network for health checking
  - Consensus mechanism for shard status (Byzantine fault tolerance)
  - Slashing for malicious oracles
- **Proof-of-Inference (PoI) Integration**: Verifiable AI outputs
  - Cryptographic signatures on all inference results
  - On-chain verification of model execution
  - Transparency for auditing AI decisions
- **Zero-Knowledge Proofs for Inference**: Privacy + verifiability
  - zkML integration (EZKL, Giza, Modulus Labs)
  - Prove inference correctness without revealing model weights
  - Enable private queries with public verifiability

#### Advanced iNFT Features
- **Dynamic Metadata Updates**: Evolving AI agents
  - On-chain update mechanism for agent memory/traits
  - Versioned metadata with rollback capability
  - Proof-of-learning for autonomous agent improvement
- **Cross-Chain iNFT Bridges**: Interoperability with other chains
  - Ethereum, Polygon, Arbitrum, Optimism bridges
  - Unified liquidity across ecosystems
  - Cross-chain AI agent execution
- **iNFT Marketplaces**: Decentralized trading platforms
  - OpenSea/Blur integration for discovery
  - Royalty enforcement for model creators
  - Renting mechanisms (temporary access to AI agents)

#### Security & Compliance
- **Audit & Bug Bounty Program**
  - Third-party smart contract audits (CertiK, OpenZeppelin, Trail of Bits)
  - Ongoing bug bounty with tiered rewards ($5K-$50K)
  - Responsible disclosure policy
- **Regulatory Compliance**: Preparing for AI governance frameworks
  - GDPR/CCPA compliance for user data handling
  - EU AI Act readiness (risk categorization, transparency)
  - KYC/AML integration for enterprise customers (optional)
- **Content Moderation**: Responsible AI deployment
  - Automated filtering for harmful outputs
  - User reporting mechanisms
  - Community governance for content policies

---

### Phase 4: Enterprise & Mass Adoption (2026+)

#### Enterprise Solutions
- **Private Deployments**: Self-hosted TeeTee instances
  - On-premises TEE infrastructure
  - Custom governance and access control
  - White-label solution for enterprises
- **B2B API Gateway**: Enterprise-grade interface
  - High-volume endpoints with guaranteed SLAs
  - Usage analytics and cost optimization tools
  - Dedicated account management
- **Hybrid Cloud Integration**: Seamless Web2/Web3 bridge
  - AWS, Azure, GCP connectors
  - Traditional authentication (OAuth, SAML) + Web3 wallets
  - Gradual onboarding path for Web2 companies

#### Mass Market Features
- **Mobile Apps**: iOS and Android native applications
  - Consumer-friendly AI chat interface
  - Push notifications for model updates
  - Biometric wallet access
- **Fiat On-Ramps**: Credit card payments for credits
  - Partner with payment providers (Stripe, Ramp, MoonPay)
  - Seamless UX without crypto knowledge
  - Automatic credit purchase and top-up
- **AI Agent Marketplace**: App store for intelligent agents
  - Browse, discover, and purchase pre-trained agents
  - One-click deployment of popular use cases
  - Rating and review system

#### Ecosystem Growth
- **DAO Governance**: Community-driven protocol evolution
  - Token-based voting on protocol upgrades
  - Treasury management for ecosystem development
  - Proposal system for feature requests
- **Grants & Accelerator**: Funding promising projects
  - $5M ecosystem fund for builders
  - Technical mentorship and marketing support
  - Success-based milestone payouts
- **Academic Partnerships**: Research collaborations
  - Joint research with universities (Stanford, MIT, Berkeley)
  - Open datasets for decentralized AI research
  - Publish findings in top-tier conferences (NeurIPS, ICML)

---

### Phase 5: Vision — Decentralized AGI Infrastructure (2027+)

#### Long-Term Moonshots
- **Fully On-Chain AI Training**: Decentralized model training at scale
  - Distributed gradient descent across 1000+ nodes
  - On-chain proof of training (blockchain-verified model provenance)
  - Incentivized participation with token rewards
- **Autonomous AI Agent Economy**: Self-sovereign AI entities
  - AI agents that own wallets and transact independently
  - DAO-governed AI collectives
  - Legal frameworks for AI personhood (collaboration with regulators)
- **Interplanetary AI Network**: Space-grade decentralized AI
  - Satellite-based TEE nodes for global coverage
  - Resilient to terrestrial disruptions
  - Low-latency inference for IoT and edge devices
- **AI Safety & Alignment Research**: Responsible AGI development
  - Open-source alignment research lab
  - Transparent evaluations of model safety
  - Collaboration with AI safety organizations (Anthropic, OpenAI, DeepMind)

---

## 🤝 Contributing

We welcome contributions from the community! TeeTee is an open-source project, and we're excited to collaborate with developers, researchers, and enthusiasts.

### How to Contribute

1. **Fork the Repository**: Click the "Fork" button on GitHub
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Your Changes**: `git commit -m 'Add some amazing feature'`
4. **Push to Your Branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**: Submit a PR with a clear description

### Contribution Guidelines

- Follow existing code style and conventions
- Write comprehensive tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR
- Use clear, descriptive commit messages

### Areas We Need Help

- 🐛 Bug fixes and performance improvements
- 📚 Documentation and tutorials
- 🧪 Test coverage expansion
- 🌐 Internationalization (i18n)
- 🎨 UI/UX enhancements
- 🔬 Research on model splitting and optimization

---

## 📞 Community & Support

### 🔗 Links

- **🌐 Live App**: [https://teetee.site](https://teetee.site)
- **📖 Documentation**: [Comprehensive Guide](https://docs.google.com/document/d/1pqDrJoYoBfVG19Kxu0-9uSHfwEq3ZQjp8d1CU9Pd-Kk/edit?usp=sharing)
- **🎥 Demo Video**: [TeeTee Wave 5 Demo](https://drive.google.com/drive/folders/1eWDgBJ_o2jr5xT2U_ZYclxhAAt15G4HJ?usp=sharing)
- **🐦 Twitter Thread**: [Building Journey & Mainnet Deployment](https://x.com/ilovedahmo/status/1986064335354126573)
- **🔍 GitHub**: [Report bugs or request features](https://github.com/your-org/TeeTee-v2/issues)

### 📜 Verified Mainnet Contracts (0G Network)

- **CreditUse**: [`0xd1ce92b2c95a892fe1166e20b65c73b33b269f7e`](https://chainscan.0g.ai/address/0xd1ce92b2c95a892fe1166e20b65c73b33b269f7e?tab=contract-viewer)
- **OracleStub**: [`0x20f8f585f8e0d3d1fce7907a3c02aeaa5c924707`](https://chainscan.0g.ai/address/0x20f8f585f8e0d3d1fce7907a3c02aeaa5c924707?tab=contract-viewer)
- **DataVerifierAdapterFixed**: [`0x8889106de495dc1731a9b60a58817de6e0142ac0`](https://chainscan.0g.ai/address/0x8889106de495dc1731a9b60a58817de6e0142ac0?tab=contract-viewer)
- **INFT (ERC-7857)**: [`0x56776a7878c7d4cc9943b17d91a3e098c77614da`](https://chainscan.0g.ai/address/0x56776a7878c7d4cc9943b17d91a3e098c77614da?tab=contract-viewer)


## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **0G Network** for providing the high-performance blockchain infrastructure
- **Phala Network** for Confidential AI API and TEE compute resources
- **0G WaveHack** for supporting our development through Wave 5
- **Open-source community** for the incredible tools and libraries

---

<div align="center">

**Built with ❤️ by the TeeTee Team**

[Website](https://teetee.site) • [Docs](https://docs.google.com/document/d/1pqDrJoYoBfVG19Kxu0-9uSHfwEq3ZQjp8d1CU9Pd-Kk/edit?usp=sharing) • [Twitter Thread](https://x.com/ilovedahmo/status/1986064335354126573) • [Demo Video](https://drive.google.com/drive/folders/1eWDgBJ_o2jr5xT2U_ZYclxhAAt15G4HJ?usp=sharing)

**Tag us**: @0G_Builders @akindo_io

</div>

