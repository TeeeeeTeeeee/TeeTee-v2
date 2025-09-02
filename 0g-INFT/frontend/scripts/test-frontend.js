/**
 * Frontend Testing Script
 * 
 * Simple test to verify contract integration and off-chain service connectivity
 * Run with: node scripts/test-frontend.js
 */

const { CONTRACT_ADDRESSES, INFT_ABI, OFFCHAIN_SERVICE_URL } = require('../lib/constants')

async function testContractIntegration() {
  console.log('🧪 Testing Frontend Integration...\n')
  
  // Test 1: Contract Addresses
  console.log('📋 Contract Addresses:')
  console.log('  INFT:', CONTRACT_ADDRESSES.INFT)
  console.log('  DataVerifier:', CONTRACT_ADDRESSES.DATA_VERIFIER)
  console.log('  Oracle:', CONTRACT_ADDRESSES.ORACLE_STUB)
  
  // Test 2: ABI Validation
  console.log('\n📄 ABI Functions Available:')
  const functions = INFT_ABI.filter(item => item.startsWith('function'))
  functions.forEach(fn => {
    console.log('  -', fn.split('(')[0].replace('function ', ''))
  })
  
  // Test 3: Off-chain Service Connectivity
  console.log('\n🌐 Testing Off-chain Service...')
  try {
    const response = await fetch(`${OFFCHAIN_SERVICE_URL}/health`)
    if (response.ok) {
      const data = await response.json()
      console.log('  ✅ Service healthy:', data.service)
      console.log('  📅 Timestamp:', data.timestamp)
    } else {
      console.log('  ❌ Service not responding:', response.status)
    }
  } catch (error) {
    console.log('  ❌ Service connection failed:', error.message)
    console.log('  💡 Make sure off-chain service is running on localhost:3000')
  }
  
  console.log('\n🎉 Frontend integration test complete!')
  console.log('💻 Start the frontend with: npm run dev')
}

// Run tests
testContractIntegration().catch(console.error)
