import { useRouter } from 'next/router';
import PillNav, { PillNavItem } from './PillNav';
import { ConnectButton } from '@rainbow-me/rainbowkit';

export const Navbar = () => {
  const router = useRouter();
  
  const navItems: PillNavItem[] = [
    { label: "Home", href: "/" },
    { label: "Chat", href: "/chat" },
    { label: "Models", href: "#" },
  ];

  // Render the RainbowKit ConnectButton
  const renderWalletButton = () => {
    return (
      <ConnectButton 
        showBalance={true}
        chainStatus="icon"
        accountStatus="address"
      />
    );
  };

  return (
    <div className="w-full" style={{ background: 'linear-gradient(to right, #a78bfa, #c4b5fd)' }}>
      <PillNav
        logo="/images/TeeTee.png"
        logoAlt="TeeTee Logo"
        items={navItems}
        activeHref={router.pathname}
        baseColor="transparent" // Transparent to show gradient background
        pillColor="#a78bfa" // Violet-400 color
        hoveredPillBgColor="#ffffff" // White background on hover
        hoveredPillTextColor="#ddd6fe" // Light violet color for text on hover
        pillTextColor="#ffffff"
        className="bg-transparent"
        initialLoadAnimation={true}
        renderWalletButton={renderWalletButton}
      />
    </div>
  );
};