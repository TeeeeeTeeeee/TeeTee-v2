import React, { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { ConnectButton } from '@rainbow-me/rainbowkit';

interface Conversation {
  id: number;
  title: string;
  date: string;
}

type NavLink = {
  href: string;
  label: string;
};

const navLinks: NavLink[] = [
  { href: '/', label: 'Home' },
  { href: '/chat', label: 'Chat' },
  { href: '/storage', label: 'Storage' },
];

// Icon components
const MenuIcon = () => (
  <motion.svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2">
    <motion.line x1="3" y1="12" x2="21" y2="12" />
  </motion.svg>
);

const XIcon = () => (
  <motion.svg
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2">
    <motion.line x1="18" y1="6" x2="6" y2="18" />
    <motion.line x1="6" y1="6" x2="18" y2="18" />
  </motion.svg>
);

const SidebarToggleIcon = () => (
  <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="7" y="7" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2"/>
    <line x1="16" y1="7" x2="16" y2="25" stroke="currentColor" strokeWidth="2"/>
  </svg>
);

interface Conversation {
  id: number;
  title: string;
  date: string;
}

const CollapsibleSection = ({ title, children }: { title: string; children: React.ReactNode }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="mb-4">
      <button
        className="w-full flex items-center justify-between py-2 px-4 rounded-xl hover:bg-gray-100"
        onClick={() => setOpen(!open)}>
        <span className="font-semibold">{title}</span>
        {open ? <XIcon /> : <MenuIcon />}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden">
            <div className="p-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Chat = () => {
  const [message, setMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('TinyLlama-1.1B-Chat-v1.0');
  const [tokenCount, setTokenCount] = useState(0);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isOpen, setIsOpen] = useState(true); // Sidebar state

  const models = [
    'TinyLlama-1.1B-Chat-v1.0 (#1)',
    'TinyLlama-1.1B-Chat-v1.0 (#2)',
    'TinyLlama-1.1B-Chat-v1.0 (#3)',
    'TinyLlama-1.1B-Chat-v1.0 (#4)',
  ];

  interface Conversation {
    id: number;
    title: string;
    date: string;
  }

  const initialConversations: Conversation[] = [
  ];

  // Initialize conversations with mock data
  React.useEffect(() => {
    setConversations(initialConversations);
  }, []);

  const deleteConversation = (id: number) => {
    setConversations(prev => prev.filter(conv => conv.id !== id));
  };

  return (
    <div className="min-h-screen bg-[#F8F9FB]">
      {/* Sidebar Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-4 z-50 md:hidden p-2 bg-white rounded-lg shadow-lg"
      >
        <motion.span
          animate={{ rotate: isOpen ? 0 : 90 }}
          transition={{ duration: 0.4, ease: 'easeInOut' }}
          style={{ display: 'inline-block' }}
        >
          <SidebarToggleIcon />
        </motion.span>
      </button>

      {/* Sidebar */}
      <motion.div 
        className="fixed left-0 top-0 bg-white border-r border-gray-200 flex flex-col h-screen"
        animate={{
          width: isOpen ? "288px" : "80px",
          x: isOpen ? 0 : -208
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {/* Logo */}
        <div className="px-4 py-4 border-b border-gray-200">
          <Link href="/" className="flex items-center gap-3">
            <Image 
              src="/images/TeeTee.png" 
              alt="TeeTee Logo" 
              width={32} 
              height={32}
              className="flex-shrink-0"
            />
            <motion.span
              animate={{
                opacity: isOpen ? 1 : 0,
                display: isOpen ? "inline" : "none",
              }}
              transition={{ duration: 0.2 }}
              className="text-xl font-['Pacifico'] bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-transparent bg-clip-text"
            >
              TeeTee
            </motion.span>
          </Link>
        </div>

        {/* Navigation Links */}
        <div className="px-4 py-4 border-b border-gray-200">
          <nav className="flex flex-col gap-2">
            {navLinks.map((link) => (
              <Link 
                key={link.href} 
                href={link.href}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors
                  ${link.href === '/chat' 
                    ? 'bg-violet-50 text-violet-600' 
                    : 'text-gray-600 hover:bg-gray-50'
                  }`}
              >
                <span className="flex-shrink-0 w-5 h-5">
                  {link.href === '/' && (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                    </svg>
                  )}
                  {link.href === '/chat' && (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                  )}
                  {link.href === '/models' && (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  )}
                </span>
                <motion.span
                  animate={{
                    opacity: isOpen ? 1 : 0,
                    display: isOpen ? "inline" : "none",
                  }}
                  transition={{ duration: 0.2 }}
                >
                  {link.label}
                </motion.span>
              </Link>
            ))}
          </nav>
        </div>

        {/* New Chat Button */}
        <div className="px-4 py-4">
          <button className="w-full bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-white px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <motion.span
              animate={{
                opacity: isOpen ? 1 : 0,
                display: isOpen ? "inline" : "none",
              }}
              transition={{ duration: 0.2 }}
            >
              New Chat
            </motion.span>
          </button>
        </div>

        {/* Model Selector */}
        <div className="px-4 mb-4">
          <div className="relative">
            <button
              onClick={() => setShowModelDropdown(!showModelDropdown)}
              className="w-full flex items-center justify-between px-4 py-2 text-sm text-gray-700 bg-white border rounded-lg hover:bg-gray-50"
            >
              <span>{selectedModel}</span>
              <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {showModelDropdown && (
              <div className="absolute w-full mt-1 bg-white border rounded-lg shadow-lg z-50">
                {models.map((model) => (
                  <button
                    key={model}
                    onClick={() => {
                      setSelectedModel(model);
                      setShowModelDropdown(false);
                    }}
                    className="w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-50 first:rounded-t-lg last:rounded-b-lg"
                  >
                    {model}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Token Counter */}
        <div className="px-4 py-2 mb-4">
          <div className="text-sm text-gray-500 flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10h-6m-2-5h10" />
            </svg>
            Tokens: {tokenCount}
          </div>
        </div>

        {/* Previous Conversations */}
        <div className="px-4 flex-1 overflow-y-auto">
          <h3 className="text-xs font-medium text-gray-500 uppercase mb-2">Previous Chats</h3>
          <div className="space-y-1">
            {conversations.length > 0 ? (
              conversations.map((conv: Conversation) => (
                <div
                  key={conv.id}
                  className="group relative"
                >
                  <button
                    className="w-full px-3 py-2 text-sm text-left text-gray-700 hover:bg-gray-50 rounded-lg flex items-center justify-between"
                  >
                    <span className="truncate flex-1">{conv.title}</span>
                    <span className="text-xs text-gray-400">{conv.date}</span>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conv.id);
                    }}
                    className="absolute right-0 top-1/2 -translate-y-1/2 mr-2 w-6 h-6 flex items-center justify-center text-gray-400 opacity-0 group-hover:opacity-100 hover:text-red-500 transition-all"
                    title="Delete conversation"
                  >
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))
            ) : (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <svg className="w-12 h-12 text-gray-300 mb-3" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <p className="text-sm text-gray-500">No conversations yet</p>
              </div>
            )}
          </div>

          {/* Connect Wallet Button */}
            <div className="absolute bottom-0 left-0 right-0 px-4 py-4 border-t border-gray-200">
              <ConnectButton.Custom>
                {({
                  account,
                  chain,
                  openAccountModal,
                  openChainModal,
                  openConnectModal,
                  mounted,
                }) => {
                  const ready = mounted;
                  const connected = ready && account && chain;

                  return (
                    <div
                      {...(!ready && {
                        'aria-hidden': true,
                        style: {
                          opacity: 0,
                          pointerEvents: 'none',
                          userSelect: 'none',
                        },
                      })}
                    >
                      {(() => {
                        if (!connected) {
                          return (
                            <button 
                              onClick={openConnectModal} 
                              className="w-full bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-white px-4 py-2 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                            >
                              <motion.span
                                animate={{
                                  opacity: isOpen ? 1 : 0,
                                  display: isOpen ? "inline" : "none",
                                }}
                                transition={{ duration: 0.2 }}
                              >
                                Connect Wallet
                              </motion.span>
                            </button>
                          );
                        }
                        
                        return (
                          <div className="flex flex-col gap-2">
                            <button
                              onClick={openChainModal}
                              className="w-full px-4 py-2 text-sm text-gray-700 bg-white border rounded-lg hover:bg-gray-50 flex items-center justify-center gap-2"
                            >
                              {chain.hasIcon && (
                                <div style={{ background: chain.iconBackground }} className="w-4 h-4 rounded-full overflow-hidden">
                                  {chain.iconUrl && (
                                    <Image
                                      alt={chain.name ?? 'Chain icon'}
                                      src={chain.iconUrl}
                                      width={16}
                                      height={16}
                                    />
                                  )}
                                </div>
                              )}
                              <motion.span
                                animate={{
                                  opacity: isOpen ? 1 : 0,
                                  display: isOpen ? "inline" : "none",
                                }}
                                transition={{ duration: 0.2 }}
                              >
                                {chain.name}
                              </motion.span>
                            </button>
                            <button
                              onClick={openAccountModal}
                              className="w-full px-4 py-2 text-sm text-gray-700 bg-white border rounded-lg hover:bg-gray-50 flex items-center justify-center gap-2"
                            >
                              <motion.span
                                animate={{
                                  opacity: isOpen ? 1 : 0,
                                  display: isOpen ? "inline" : "none",
                                }}
                                transition={{ duration: 0.2 }}
                              >
                                {account.displayName}
                              </motion.span>
                            </button>
                          </div>
                        );
                      })()}
                    </div>
                  );
                }}
              </ConnectButton.Custom>
            </div>
        </div>
      </motion.div>

      {/* Main Chat Area */}
      <motion.main 
        className="h-screen flex flex-col"
        animate={{
          paddingLeft: isOpen ? "288px" : "80px"
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {/* Desktop Sidebar Toggle */}
        <div className="hidden md:block pt-6 px-6">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center justify-center"
            title={isOpen ? "Close Sidebar" : "Open Sidebar"}
          >
            <motion.span
              animate={{ rotate: isOpen ? 0 : 90 }}
              transition={{ duration: 0.4, ease: 'easeInOut' }}
              style={{ display: 'inline-block' }}
            >
              <SidebarToggleIcon />
            </motion.span>
          </button>
        </div>

        <div className="flex-1 px-6 overflow-y-auto">
          <div className="max-w-3xl mx-auto">
            {/* Welcome Message */}
            <div className="text-center py-20">
              <h1 className="text-4xl font-bold mb-4">
                Welcome to <span className="text-violet-400">TeeTee</span>
              </h1>
              <p className="text-gray-600 mb-12">
                This is a secure AI assistant running in a TEE
              </p>

              {/* Feature Icons */}
              <div className="flex justify-center gap-8">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-lg bg-violet-100 flex items-center justify-center">
                    <svg className="w-6 h-6 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <span className="text-sm">Reasoning</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-lg bg-violet-100 flex items-center justify-center">
                    <svg className="w-6 h-6 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <span className="text-sm">Create Image</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 rounded-lg bg-violet-100 flex items-center justify-center">
                    <svg className="w-6 h-6 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <span className="text-sm">Deep Research</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Input */}
        <div className="border-t border-gray-200 bg-white px-6 py-4">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <button 
                className="absolute left-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-violet-400 transition-colors"
                title="Attach file"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              </button>
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Initiate a query or send a command to the AI..."
                className="w-full pl-12 pr-12 py-3 rounded-lg border border-gray-200 focus:outline-none focus:border-violet-400"
              />
              <button 
                className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-violet-400 text-white rounded-lg hover:bg-violet-500 transition-colors"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </div>
            <p className="text-center text-xs text-gray-500 mt-2">
              All responses are verified through decentralized TEE computation
            </p>
          </div>
        </div>
      </motion.main>
    </div>
  );
};

export default Chat;