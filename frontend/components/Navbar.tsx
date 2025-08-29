import Image from 'next/image';

export const Navbar = () => {
  return (
    <nav className="fixed top-0 w-full px-6 py-4 bg-white/80 backdrop-blur-sm z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Image 
            src="/images/TeeTee.png" 
            alt="TeeTee Logo" 
            width={48} 
            height={48}
            priority
          />
          <span className="text-2xl font-['Pacifico'] bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-transparent bg-clip-text">
            TeeTee
          </span>
        </div>
        
        <div className="flex-1 flex justify-center gap-12">
          <a href="/chat" className="text-black hover:text-gray-600 transition-colors text-sm font-medium">Chat</a>
          <a href="#" className="text-black hover:text-gray-600 transition-colors text-sm font-medium">Models</a>
        </div>

        <button className="bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-white px-6 py-2 rounded-full text-sm font-medium hover:opacity-90 transition-opacity">
          Get Started
        </button>
      </div>
    </nav>
  );
};