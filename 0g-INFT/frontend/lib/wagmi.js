import { createConfig, http } from 'wagmi'
import { injected, walletConnect } from 'wagmi/connectors'
import { ZERO_G_NETWORK } from './constants'

// Create the wagmi config for 0G Galileo testnet
export const config = createConfig({
  chains: [ZERO_G_NETWORK],
  connectors: [
    injected(),
    // WalletConnect can be added later if needed
    // walletConnect({ projectId: 'your-project-id' }),
  ],
  transports: {
    [ZERO_G_NETWORK.id]: http('https://evmrpc-testnet.0g.ai'),
  },
})

// Utility function to add 0G network to user's wallet
export const addZeroGNetwork = async () => {
  try {
    await window.ethereum.request({
      method: 'wallet_addEthereumChain',
      params: [
        {
          chainId: `0x${ZERO_G_NETWORK.id.toString(16)}`,
          chainName: ZERO_G_NETWORK.name,
          nativeCurrency: ZERO_G_NETWORK.nativeCurrency,
          rpcUrls: ZERO_G_NETWORK.rpcUrls.default.http,
          blockExplorerUrls: [ZERO_G_NETWORK.blockExplorers.default.url],
        },
      ],
    })
    return true
  } catch (error) {
    console.error('Failed to add 0G network:', error)
    return false
  }
}
