import React from 'react';

export const Hero = () => {
  return (
    <section className="pt-32 pb-24">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col items-center text-center">
          {/* Brand Name */}
          <h1 className="text-[72px] font-bold bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-transparent bg-clip-text mb-0">
            TeeTee
          </h1>

          {/* Main Title */}
          <div className="mb-2">
            <h2 className="text-[48px] font-bold text-gray-900">
              Verifiable, Decentralized
            </h2>
            <h2 className="text-[48px] font-bold text-violet-400">
              AI Inference
            </h2>
          </div>

          {/* Description */}
          <p className="text-xl text-gray-600 max-w-3xl mb-12">
            Sharding large language models across Trusted Execution Environments
            to deliver secure, scalable, and verifiable AI inference with on-chain attestations.
          </p>

          {/* Call to Action Buttons */}
          <div className="flex gap-6">
            <button className="bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-white px-8 py-3 rounded-full text-lg font-medium hover:opacity-90 transition-opacity">
              Join Early Access
            </button>
            <button className="border-2 border-violet-400 text-violet-400 px-8 py-3 rounded-full text-lg font-medium hover:bg-violet-50 transition-colors">
              Read Whitepaper
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};