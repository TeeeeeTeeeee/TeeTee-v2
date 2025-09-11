import '@rainbow-me/rainbowkit/styles.css';
import React, { PropsWithChildren } from 'react';
import { getDefaultConfig, RainbowKitProvider, type Chain } from '@rainbow-me/rainbowkit';
import { WagmiProvider } from 'wagmi';
import {
  mainnet,
  polygon,
  optimism,
  arbitrum,
  base,
} from 'wagmi/chains';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';

// Define custom chains with proper typing
const galileo = {
  id: 16601,
  name: '0G-Galileo-Testnet',
  iconUrl: '/0g.webp',
  iconBackground: '#fff',
  nativeCurrency: { name: '0G', symbol: 'OG', decimals: 18 },
  rpcUrls: {
    default: { http: ['https://evmrpc-testnet.0g.ai/'] },
  },
  blockExplorers: {
    default: { name: '0G-Galileo-Testnet', url: 'https://chainscan-galileo.0g.ai' },
  },
} as const satisfies Chain;

const projectId = process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID ?? 'YOUR_PROJECT_ID';

const config = getDefaultConfig({
  appName: 'My RainbowKit App',
  projectId,
  chains: [galileo],
  ssr: true, // If your dApp uses server side rendering (SSR)
});

const queryClient = new QueryClient();

export function Providers({ children }: PropsWithChildren) {
  return (
    <WagmiProvider config={config}>
      <QueryClientProvider client={queryClient}>
        <RainbowKitProvider>
          {children}
        </RainbowKitProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
}

export default Providers;