import { useRouter } from 'next/router';
import PillNav, { PillNavItem } from './PillNav';
import { ConnectButton } from '@rainbow-me/rainbowkit';

export const Navbar = () => {
  const router = useRouter();
  
  const navItems: PillNavItem[] = [
    { label: "Home", href: "/" },
    { label: "Chat", href: "/chat" },
    { label: "Models", href: "#" },
    { label: "Storage", href: "/storage" },
  ];

  // Render the RainbowKit ConnectButton
  const renderWalletButton = () => {
    return (
      <ConnectButton 
        showBalance={false}
        chainStatus="icon"
        accountStatus="address"
      />
    );
  };

  return (
    <div className="w-full h-[40px]">
      <PillNav
        logo="/images/TeeTee.png"
        logoAlt="TeeTee Logo"
        items={navItems}
        activeHref={router.pathname}
        baseColor="#ffffff" // Pure white background
        pillColor="#7c3aed" // Violet color
        hoveredPillTextColor="#ddd6fe" // Light violet color
        pillTextColor="#ffffff"
        className="bg-white"
        initialLoadAnimation={true}
        renderWalletButton={renderWalletButton}
      />
    </div>
  );
};