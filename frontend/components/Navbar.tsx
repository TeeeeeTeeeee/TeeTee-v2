import { Home, MessageSquare, Box, HelpCircle } from 'lucide-react';
import { NavBar } from './ui/tubelight-navbar';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useState, useEffect } from 'react';

export const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  const navItems = [
    { name: 'Home', url: '/', icon: Home },
    { name: 'Chat', url: '/chat', icon: MessageSquare },
    { name: 'Models', url: '/models', icon: Box },
    { name: 'FAQ', url: '/faq', icon: HelpCircle },
  ];

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div 
      className={`fixed top-0 left-0 w-full z-[1000] py-4 transition-colors duration-300 ${
        isScrolled ? 'bg-white/90 backdrop-blur-sm shadow-sm' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 w-full relative flex items-center justify-between">
        {/* TeeTee Logo on the left - aligned with navbar */}
        <div className="flex items-center z-20 h-12">
          <h1 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-pacifico)' }}>
            <span className="bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text">
              TeeTee
            </span>
          </h1>
        </div>
        
        {/* Tubelight Navbar in the center (absolute positioning to keep it centered) */}
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