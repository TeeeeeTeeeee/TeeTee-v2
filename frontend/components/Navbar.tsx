import { Home, MessageSquare, Box, HelpCircle } from 'lucide-react';
import { NavBar } from './ui/tubelight-navbar';
import { ConnectButton } from '@rainbow-me/rainbowkit';

export const Navbar = () => {
  const navItems = [
    { name: 'Home', url: '/', icon: Home },
    { name: 'Chat', url: '/chat', icon: MessageSquare },
    { name: 'Models', url: '/models', icon: Box },
    { name: 'FAQ', url: '/faq', icon: HelpCircle },
  ];

  return (
    <div className="fixed top-0 left-0 w-full z-[1000] py-4">
      <div className="max-w-7xl mx-auto px-6 w-full relative flex items-center justify-between">
        {/* Logo on the left */}
        <div className="z-20">
          <img 
            src="/images/TeeTee.png" 
            alt="TeeTee Logo" 
            className="w-12 h-12 object-cover rounded-full"
          />
        </div>
        
        {/* Tubelight Navbar in the center */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <NavBar items={navItems} className="relative" />
        </div>
        
        {/* Wallet button on the right */}
        <div className="z-20">
          <ConnectButton 
            showBalance={true}
            chainStatus="icon"
            accountStatus="address"
          />
        </div>
      </div>
    </div>
  );
};