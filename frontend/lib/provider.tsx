import '@rainbow-me/rainbowkit/styles.css';
import React, { PropsWithChildren } from 'react';
import { getDefaultConfig, RainbowKitProvider } from '@rainbow-me/rainbowkit';
import { WagmiProvider } from 'wagmi';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { galileoTestnet, zgMainnet, getCurrentChain, NETWORK_TYPE } from './networkConfig';

const projectId = process.env.NEXT_PUBLIC_WALLETCONNECT_PROJECT_ID ?? 'YOUR_PROJECT_ID';

// Get the current chain based on environment configuration
const currentChain = getCurrentChain();

// Configure wagmi with the appropriate chain
const config = getDefaultConfig({
  appName: 'TeeTee - 0G Decentralized AI',
  projectId,
  chains: [currentChain],
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

// Export network info for debugging
export const networkInfo = {
  current: NETWORK_TYPE,
  chain: currentChain,
};

export default Providers;