import React from 'react';
import { ConnectButton } from '@rainbow-me/rainbowkit';

// A simple, reusable header that places RainbowKit's Connect Wallet button on the right
export default function Header() {
  return (
    <header className="w-full border-b border-black/[.08] dark:border-white/[.145] relative z-10">
      <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
        <div className="text-base sm:text-lg font-semibold tracking-tight">My dApp</div>
        <ConnectButton />
      </div>
    </header>
  );
}
