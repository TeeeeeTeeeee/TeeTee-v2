#!/bin/bash

# Quick test script for LLM server
# Usage: ./quick-test.sh

echo "🧪 Testing LLM Server..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please create .env file with your PHALA_API_KEY"
    exit 1
fi

# Check if API key is set
if ! grep -q "PHALA_API_KEY=.*[a-zA-Z0-9]" .env; then
    echo -e "${YELLOW}⚠️  PHALA_API_KEY not set in .env file${NC}"
    echo "Please add your API key to the .env file"
    exit 1
fi

echo "1️⃣  Testing Health Endpoint..."
HEALTH=$(curl -s http://localhost:3001/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
    echo "$HEALTH" | jq . 2>/dev/null || echo "$HEALTH"
else
    echo -e "${RED}❌ Server not responding. Is it running?${NC}"
    echo "Start server with: npm start"
    exit 1
fi

echo ""
echo "2️⃣  Testing Models Endpoint..."
MODELS=$(curl -s http://localhost:3001/models)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Models endpoint working${NC}"
    echo "$MODELS" | jq . 2>/dev/null || echo "$MODELS"
else
    echo -e "${RED}❌ Models endpoint failed${NC}"
    exit 1
fi

echo ""
echo "3️⃣  Testing Inference Endpoint..."
INFERENCE=$(curl -s -X POST http://localhost:3001/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say hello in one word",
    "model": "deepseek-v3"
  }')

if [ $? -eq 0 ] && echo "$INFERENCE" | grep -q "success"; then
    echo -e "${GREEN}✅ Inference test passed${NC}"
    echo "$INFERENCE" | jq . 2>/dev/null || echo "$INFERENCE"
else
    echo -e "${RED}❌ Inference test failed${NC}"
    echo "Response: $INFERENCE"
    echo ""
    echo "Common issues:"
    echo "- API key not valid (check Phala dashboard)"
    echo "- Insufficient balance (need at least \$5)"
    echo "- Model not available (try: qwen-2.5-7b)"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 All tests passed! Server is ready to use.${NC}"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3000/testing in your browser"
echo "  2. Start chatting with the LLM"
echo ""

