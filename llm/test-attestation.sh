#!/bin/bash

# Test script for TEE Attestation endpoints
# Usage: ./test-attestation.sh

SERVER_URL="${1:-http://localhost:3001}"

echo "🧪 Testing TEE Attestation Endpoints"
echo "Server: $SERVER_URL"
echo "="

====================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "1️⃣  Testing Health Check..."
curl -s "$SERVER_URL/health" | jq .
echo ""

echo "2️⃣  Testing Quick Attestation (GET)..."
curl -s "$SERVER_URL/attest/quick" | jq .
echo ""

echo "3️⃣  Testing Custom Data Attestation (POST)..."
curl -s -X POST "$SERVER_URL/attest" \
  -H "Content-Type: application/json" \
  -d '{
    "userData": "my-custom-data-for-attestation"
  }' | jq .
echo ""

echo "4️⃣  Testing TEE Info..."
curl -s "$SERVER_URL/tee/info" | jq .
echo ""

echo "5️⃣  Testing Attested Inference..."
curl -s -X POST "$SERVER_URL/inference/attested" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "model": "deepseek-v3",
    "includeAttestation": true
  }' | jq .
echo ""

echo -e "${GREEN}✅ All attestation endpoint tests completed!${NC}"
echo ""
echo "Note: If running locally, you'll see mock data."
echo "      Deploy to Phala TEE for real attestation."

