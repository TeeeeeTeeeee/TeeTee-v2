"use client";

import TextPressure from './TextPressure';

export const Footer = () => {
  return (
    <footer className="py-16 px-6 bg-gradient-to-r from-violet-100/40 via-purple-100/30 to-violet-100/40">
        <div className="max-w-7xl mx-auto">
          {/* Large Brand Text with TextPressure at the top */}
          <div className="mb-10">
            <div style={{ position: 'relative', height: '290px', width: '100%', padding: '0', overflow: 'visible' }}>
              <TextPressure
                text="TEETEE"
                flex={false} /* Set to false to allow custom positioning */
                alpha={false}
                stroke={false}
                width={true}
                weight={true}
                italic={true}
                textColor="linear-gradient(to right, #a78bfa, #d8b4fe)"
                strokeColor="#ddd6fe" 
                fontFamily="Geist, Inter, sans-serif"
                fontUrl=""
                minFontSize={40} // Increased font size
                className="w-full"
              />
            </div>
          </div>

          {/* Contact and Social Media Grid */}
          <div className="grid grid-cols-4 gap-20 mb-4">
            {/* Introduction Section */}
            <div className="col-span-2">
              <h3 className="text-2xl font-bold mb-0 leading-relaxed">
                Revolutionizing AI inference through decentralized TEE networks. Secure, scalable, and verifiable computation for the future
              </h3>
            </div>

            {/* Developers Section */}
            <div>
              <h3 className="text-xl font-medium mb-0">Developers</h3>
              <div className="space-y-2">
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">Documentation</a>
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">API Reference</a>
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">Community</a>
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">SDK Downloads</a>
              </div>
            </div>

            {/* Follow Section */}
            <div>
              <h4 className="text-xl font-medium mb-0">Follow</h4>
              <div className="space-y-2">
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">↗ X</a>
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">↗ GitHub</a>
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">↗ Discord</a>
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">↗ LinkedIn</a>
              </div>
            </div>
          </div>
            
          {/* Horizontal line with copyright and policies */}
          <div className="border-t flex justify-between items-center pt-4 mt-10">
            <p className="text-sm">©TeeTee</p>
            <p className="text-sm">POLICIES</p>
          </div>
        </div>
    </footer>
  );
};
