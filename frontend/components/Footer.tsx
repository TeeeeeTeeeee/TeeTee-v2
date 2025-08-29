export const Footer = () => {
  return (
    <footer className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
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
                <a href="#" className="block text-lg text-gray-600 hover:text-gray-900">↗ Instagram</a>
              </div>
            </div>
          </div>

          {/* Large Brand Text */}
          <div className="text-[200px] font-bold leading-none mb-0">
            TEETEE
          </div>

          {/* Bottom Bar */}
          <div className="pt-8 border-t flex justify-between items-center">
            <p className="text-sm">©TeeTee</p>
            <p className="text-sm">POLICIES</p>
          </div>
        </div>
    </footer>
  );
};