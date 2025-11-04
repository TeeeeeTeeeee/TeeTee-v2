import { useRouter } from 'next/router';
import PillNav, { PillNavItem } from './PillNav';
import { ConnectButton } from '@rainbow-me/rainbowkit';

interface NavbarProps {
  sidebarExpanded?: boolean; // Is the sidebar expanded or collapsed
}

export const Navbar = ({ sidebarExpanded = false }: NavbarProps) => {
  const router = useRouter();
  
  const navItems: PillNavItem[] = [
    { label: "Home", href: "/" },
    { label: "Chat", href: "/chat" },
    { label: "Models", href: "/models" },
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

  // Dynamic leftOffset based on sidebar state
  const leftOffset = sidebarExpanded ? "650px" : "580px";

  return (
    <div className="w-full" style={{ background: 'linear-gradient(to right, #a78bfa, #c4b5fd)' }}>
      <PillNav
        items={navItems}
        activeHref={router.pathname}
        baseColor="transparent" // Transparent to show gradient background
        pillColor="#a78bfa" // Violet-400 color
        hoveredPillBgColor="#ffffff" // White background on hover
        hoveredPillTextColor="#ddd6fe" // Light violet color for text on hover
        pillTextColor="#ffffff"
        className="bg-transparent"
        initialLoadAnimation={false}
        renderWalletButton={renderWalletButton}
        leftOffset={leftOffset} // Dynamically changes: 650px when expanded, 580px when collapsed
        maxWidth="1400px" // Width extends more to the right (adjust as needed)
        align="left" // Align to left side to decrease left width, increase right width
      />
    </div>
  );
};