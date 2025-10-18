#!/bin/bash

# 0G INFT Deployment Script
# Usage: ./deploy.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🚀 0G INFT Complete Deployment to Galileo Testnet       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create .env file with your PRIVATE_KEY"
    exit 1
fi

# Check if node_modules exists
if [ ! -d node_modules ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    npm install
fi

# Compile contracts
echo -e "${YELLOW}🔨 Compiling contracts...${NC}"
npm run build

# Run deployment
echo -e "\n${YELLOW}🚀 Deploying contracts to Galileo testnet...${NC}"
npx hardhat run scripts/deploy-all.ts --network galileo

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Deployment successful!${NC}"
    
    # Update .env files
    echo -e "\n${YELLOW}🔄 Updating .env files...${NC}"
    npx hardhat run scripts/update-env.ts
    
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✨ All done! Your contracts are deployed and configured.${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    
    echo -e "\n📋 Next steps:"
    echo "   1. Check deployment details in: deployments/galileo.json"
    echo "   2. Update frontend constants if needed"
    echo "   3. Start services:"
    echo "      cd offchain-service && npm start"
    echo "      cd frontend && npm run dev"
    echo ""
    echo "   4. Test minting:"
    echo "      npx hardhat run scripts/mint.ts --network galileo"
    
else
    echo -e "\n${RED}❌ Deployment failed!${NC}"
    echo "Check the error messages above and:"
    echo "   - Ensure your wallet has enough 0G tokens"
    echo "   - Visit: https://faucet.0g.ai"
    echo "   - Check network connectivity"
    exit 1
fi

