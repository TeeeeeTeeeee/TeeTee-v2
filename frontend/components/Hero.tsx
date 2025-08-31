"use client";

import React, { useState, useEffect } from 'react';
import Particles from './Particles';

export const Hero = () => {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);
  
  return (
    <section className="relative min-h-[80vh]">
      {/* Particles background - full-size container with absolute position 
          Using z-0 to position it behind content but without fixed position
          to allow proper mouse interaction */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        {isMounted && (
          <Particles
            particleColors={['#a78bfa', '#c4b5fd', '#ddd6fe']}
            particleCount={200}
            particleSpread={10}
            speed={0.1}
            particleBaseSize={100}
            moveParticlesOnHover={true}
            particleHoverFactor={1.5}
            alphaParticles={true}
            disableRotation={false}
            className="w-full h-full"
          />
        )}
      </div>
      
      {/* Content container with appropriate padding and transparent background 
          Using pointer-events-none to allow mouse events to pass through to the particles */}
      <div className="pt-32 pb-16 flex items-center relative z-10 pointer-events-none">
        <div className="max-w-7xl mx-auto px-6 w-full">
          <div className="flex flex-col items-center text-center">
            {/* Brand Name */}
            <h1 className="text-[72px] font-bold bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-transparent bg-clip-text mb-0 pointer-events-none">
              TeeTee
            </h1>

            {/* Main Title */}
            <div className="mb-2 pointer-events-none">
              <h2 className="text-[48px] font-bold text-gray-900">
                Verifiable, Decentralized
              </h2>
              <h2 className="text-[48px] font-bold text-violet-400">
                AI Inference
              </h2>
            </div>

            {/* Description */}
            <p className="text-xl text-gray-600 max-w-3xl mb-12 pointer-events-none">
              Sharding large language models across Trusted Execution Environments
              to deliver secure, scalable, and verifiable AI inference with on-chain attestations.
            </p>

            {/* Call to Action Buttons - Using pointer-events-auto to ensure buttons are clickable */}
            <div className="flex gap-6">
              <button className="pointer-events-auto bg-gradient-to-r from-violet-400 via-violet-200 to-purple-300 text-white px-8 py-3 rounded-full text-lg font-medium hover:opacity-90 transition-opacity">
                Join Early Access
              </button>
              <button className="pointer-events-auto border-2 border-violet-400 text-violet-400 px-8 py-3 rounded-full text-lg font-medium hover:bg-violet-50 transition-colors">
                Read Whitepaper
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};