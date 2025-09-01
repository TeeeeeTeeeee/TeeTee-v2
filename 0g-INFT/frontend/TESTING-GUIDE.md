# 0G INFT Frontend Testing Guide

## Prerequisites Setup

### 1. Environment Verification
Before testing, ensure all services are running:

#### A. Check Off-chain Service Status
```bash
# In terminal 1 - Navigate to project root
cd /Users/marcus/Projects/0g-INFT

# Check if off-chain service is running
curl http://localhost:3000/health

# If not running, start it:
cd offchain-service
npm start
```

Expected response:
```json
{
  "status": "healthy",
  "service": "0G INFT Off-Chain Inference Service",
  "timestamp": "2025-01-XX..."
}
```

#### B. Start Frontend Development Server
```bash
# In terminal 2 - Navigate to frontend
cd /Users/marcus/Projects/0g-INFT/frontend

# Start the development server
npm run dev
```

Expected output:
```
✓ Ready in 2.1s
○ Local:        http://localhost:3000
○ Environments: .env.local
```

### 2. Wallet Setup

#### A. Install MetaMask
- Download and install MetaMask browser extension
- Create or import a wallet
- Make sure you have the recovery phrase safely stored

#### B. Get Test Funds
1. Visit [0G Faucet](https://faucet.0g.ai)
2. Request test tokens for your wallet address
3. Wait for tokens to arrive (usually 1-2 minutes)

## Step-by-Step Testing Process

### Phase 1: Initial Connection

#### Step 1.1: Access the Application
1. Open browser and navigate to `http://localhost:3000`
2. **Expected**: See the wallet connection screen with "Connect Wallet" button
3. **Verify**: Page loads without console errors

#### Step 1.2: Connect Wallet
1. Click "Connect Wallet" button
2. **Expected**: MetaMask popup appears
3. Select your wallet account
4. Click "Connect"
5. **Expected**: MetaMask may prompt to add 0G Galileo network
6. Click "Add Network" if prompted
7. **Expected**: Dashboard loads showing your connected account

#### Step 1.3: Verify Network Connection
1. **Check**: Top right shows "Connected Account" with your address
2. **Check**: Network status shows "0G Galileo Testnet Active"
3. **Check**: Balance shows "0 INFTs" (initially)
4. **Check**: Next Token ID shows a number (likely 2 if token 1 exists)

### Phase 2: Read-Only Operations Testing

#### Step 2.1: Dashboard Status Verification
1. **Verify**: "Next Token ID" displays correctly
2. **Verify**: "My INFTs" shows 0 (for new wallets)
3. **Verify**: "0G Galileo" shows "Testnet Active"
4. **No Action Required** - Just verify data loads

#### Step 2.2: Form Validation
1. Try submitting empty forms in each section
2. **Expected**: Browser shows "Please fill all fields" alert
3. **Verify**: No transactions are sent for invalid inputs

### Phase 3: Authorization Testing

#### Step 3.1: Test Authorization Function
**Prerequisites**: There should be an existing token (ID 1) owned by the contract deployer

1. **Fill Authorize Usage form**:
   - Token ID: `1`
   - User Address: `[YOUR_WALLET_ADDRESS]`

2. **Click "Authorize User"**
3. **Expected**: Transaction will fail with "Access denied" or similar
   - This is expected because you're not the token owner
   - This confirms the authorization system is working

#### Step 3.2: Check Authorization Status
1. Open browser console (F12)
2. This step verifies the read functions work correctly

### Phase 4: Inference Testing

#### Step 4.1: Test Unauthorized Inference
1. **Fill AI Inference form**:
   - Token ID: `1`
   - Input Prompt: `inspire me`

2. **Click "Run Inference"**
3. **Expected**: Error message appears saying user is not authorized
   - This confirms authorization checking works

#### Step 4.2: Test with Authorized User
**Note**: If you have access to the contract owner wallet, you can authorize yourself first

1. **If you're the contract owner**:
   - Use Authorize Usage to grant yourself access
   - Then retry inference
   - **Expected**: Inference succeeds and shows a quote

2. **If you're not the contract owner**:
   - This step will fail as expected
   - This confirms the security model works

### Phase 5: Mint Testing (Owner Only)

#### Step 5.1: Test Mint Function
**Note**: Only the contract owner can mint new tokens

1. **Fill Mint INFT form**:
   - Recipient Address: `[YOUR_WALLET_ADDRESS]`
   - Encrypted URI: `0g://storage/test123`
   - Metadata Hash: `0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef`

2. **Click "Mint INFT"**
3. **Expected**: 
   - If you're the owner: Transaction succeeds
   - If you're not the owner: Transaction fails with "Access denied"

### Phase 6: Transaction Flow Testing

#### Step 6.1: Monitor Transaction Status
When performing any transaction:

1. **Check**: Button shows "Processing..." or similar loading state
2. **Check**: Transaction status card appears at bottom
3. **Check**: Shows progression: "Pending..." → "Waiting for confirmation..." → "Confirmed!"
4. **Check**: Transaction hash is displayed
5. **Check**: Success message appears

#### Step 6.2: Test Error Handling
1. **Try**: Reject a transaction in MetaMask
2. **Expected**: Error message displays clearly
3. **Check**: Interface returns to normal state after error

### Phase 7: Network and Connection Testing

#### Step 7.1: Test Network Switching
1. **In MetaMask**: Switch to a different network (e.g., Ethereum Mainnet)
2. **Expected**: Dashboard may show connection issues
3. **Switch back**: To 0G Galileo testnet
4. **Expected**: Dashboard functions normally again

#### Step 7.2: Test Disconnect/Reconnect
1. **Click "Disconnect"** button
2. **Expected**: Returns to wallet connection screen
3. **Reconnect**: Click "Connect Wallet" again
4. **Expected**: Dashboard loads normally

## Expected Test Results Summary

### ✅ Successful Test Outcomes

| Test | Expected Result | Status |
|------|----------------|--------|
| Page Load | Dashboard displays without errors | ✅ |
| Wallet Connection | MetaMask connects successfully | ✅ |
| Network Addition | 0G Galileo added automatically | ✅ |
| Data Loading | Contract data displays correctly | ✅ |
| Authorization Check | Unauthorized inference blocked | ✅ |
| Form Validation | Empty forms show error messages | ✅ |
| Transaction Flow | Status updates correctly | ✅ |
| Error Handling | Clear error messages display | ✅ |

### ❌ Expected Failures (Security Working)

| Test | Expected Failure | Reason |
|------|------------------|--------|
| Mint (Non-owner) | Transaction fails | Only owner can mint |
| Authorize (Non-owner) | Transaction fails | Only token owner can authorize |
| Inference (Unauthorized) | Service rejects request | Authorization required |

## Troubleshooting Common Issues

### Issue 1: MetaMask Not Connecting
**Solutions**:
- Refresh page and try again
- Unlock MetaMask wallet
- Check if MetaMask is installed properly
- Try a different browser

### Issue 2: Network Not Added
**Solutions**:
- Manually add 0G Galileo testnet:
  - Network Name: `0G Galileo Testnet`
  - RPC URL: `https://evmrpc-testnet.0g.ai`
  - Chain ID: `16601`
  - Currency Symbol: `0G`
  - Explorer: `https://chainscan-galileo.0g.ai`

### Issue 3: Off-chain Service Connection Failed
**Solutions**:
- Check if service is running: `curl http://localhost:3000/health`
- Restart the service: `cd offchain-service && npm start`
- Check firewall settings

### Issue 4: No Test Tokens
**Solutions**:
- Visit [0G Faucet](https://faucet.0g.ai) to get test tokens
- Wait a few minutes for tokens to arrive
- Check your wallet address is correct

### Issue 5: Transaction Fails
**Solutions**:
- Check you have sufficient 0G tokens for gas
- Verify you're on the correct network
- Check the contract addresses in console

## Advanced Testing (Optional)

### Developer Testing
1. **Open Browser Console** (F12)
2. **Check for errors** in console logs
3. **Monitor network requests** in Network tab
4. **Verify contract calls** are being made correctly

### Contract Interaction Testing
```javascript
// In browser console, test direct contract interaction
console.log('Contract Address:', window.ethereum)
```

## Test Completion Checklist

- [ ] Frontend loads without errors
- [ ] Wallet connects successfully
- [ ] 0G network added automatically
- [ ] Dashboard displays correct data
- [ ] Authorization system works (blocks unauthorized users)
- [ ] Inference works for authorized users
- [ ] Transaction status updates correctly
- [ ] Error messages are clear and helpful
- [ ] Disconnect/reconnect works properly
- [ ] Forms validate inputs correctly

## Success Criteria

**The frontend passes testing if**:
1. All wallet connection flows work smoothly
2. Contract data loads and displays correctly
3. Authorization system properly blocks unauthorized access
4. Authorized inference returns AI-generated quotes
5. Transaction status tracking works reliably
6. Error handling provides clear feedback
7. User interface is responsive and intuitive

## Next Steps After Testing

1. **Production Deployment**: If all tests pass, the frontend is ready for deployment
2. **User Feedback**: Gather feedback from actual users
3. **Performance Optimization**: Monitor loading times and optimize if needed
4. **Feature Enhancement**: Add requested features based on user needs

---

**Note**: This testing guide assumes you have the complete backend infrastructure running (contracts deployed, off-chain service operational). The frontend is designed to work seamlessly with the existing Phase 1-7 implementations.
