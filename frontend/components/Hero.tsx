"use client";

import React, { useState, useEffect } from 'react';
import Particles from './Particles';

export const Hero = () => {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);
  
  return (
    <section className="relative min-h-screen bg-transparent">
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
      <div className="pt-48 pb-24 flex items-center relative z-10 pointer-events-none">
        <div className="max-w-7xl mx-auto px-6 w-full">
          <div className="flex flex-col items-center text-center">
            {/* Main Title */}
            <div className="mb-2 pointer-events-none">
              <h2 className="text-[38px] font-bold text-gray-900">
                This isn't just infrastructure.
              </h2>
              <h2 className="text-[48px] font-bold text-black mb-4">
                This is <span style={{
                  background: 'linear-gradient(to right, #a78bfa, #d8b4fe)',
                  WebkitBackgroundClip: 'text',
                  backgroundClip: 'text',
                  color: 'transparent',
                  fontSize: 64
                }}>TeeTee</span> — AI inference, reimagined for trust, scale, and resilience.
              </h2>
            </div>

            {/* Description */}
            <p className="text-xl text-gray-600 max-w-5xl mb-8 pointer-events-none">
              By sharding models across a decentralized TEE network, TeeTee removes the limits of cost, privacy, and scale. 
              It's AI infrastructure without vendor lock-in — verifiable, resilient, and open to all.
            </p>

            {/* Call to Action Buttons - Using pointer-events-auto to ensure buttons are clickable */}
            <div className="flex gap-6">
              <button className="pointer-events-auto bg-gradient-to-r from-violet-400 to-purple-300 text-white px-8 py-3 rounded-full text-lg font-medium hover:opacity-90 transition-opacity">
                Run on TeeTee
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};