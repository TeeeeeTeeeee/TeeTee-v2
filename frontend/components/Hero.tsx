"use client";

import React, { useState, useEffect } from 'react';
import Spline from '@splinetool/react-spline';
import { useRouter } from 'next/router';

export const Hero = () => {
  const [isMounted, setIsMounted] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setIsMounted(true);
  }, []);
  
  return (
    <section className="relative h-screen bg-transparent z-0">
      {/* Content container - Grid layout with text on left, Spline on right */}
      <div className="pt-32 pb-16 flex items-center relative z-10 h-full">
        <div className="max-w-7xl mx-auto px-6 w-full">
          <div className="grid grid-cols-2 gap-12 items-center">
            {/* Left Side - Text Content */}
            <div className="flex flex-col text-left pointer-events-none">
              {/* Main Title */}
              <div className="mb-6">
                <h1 className="text-[120px] font-bold leading-none mb-6" style={{ fontFamily: 'var(--font-pacifico)' }}>
                  <span className="bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text">
                    TeeTee
                  </span>
                </h1>
                <h2 className="text-[28px] font-medium text-gray-700 mb-4 leading-snug">
                LLM Sharding Across Multiple Verifiable TEE with Decentralized Inference
                </h2>
              </div>

              {/* Description */}
              <p className="text-[17px] text-gray-600 mb-8 max-w-xl leading-relaxed">
                By sharding models across a decentralized TEE network, TeeTee removes the limits of cost, privacy, and scale. Any company can host their own powerful LLM with full data privacy at half the cost or lesser.
              </p>

              {/* Call to Action Buttons - Using pointer-events-auto to ensure buttons are clickable */}
              <div className="flex gap-6">
                <button 
                  onClick={() => router.push('/chat')}
                  className="pointer-events-auto bg-gradient-to-r from-violet-400 to-purple-300 text-white px-8 py-3 rounded-full text-lg font-medium hover:opacity-90 transition-opacity"
                >
                  Run on TeeTee
                </button>
              </div>
            </div>

            {/* Right Side - Spline 3D Scene */}
            <div className="h-[600px] w-full relative overflow-visible" style={{ transform: 'scale(1.2)' }}>
              {isMounted && (
                <div style={{ 
                  width: '100%', 
                  height: '100%',
                  mixBlendMode: 'multiply',
                  background: 'transparent'
                }}>
                  <Spline
                    scene="https://prod.spline.design/rUhzSyDm4oYYFG4p/scene.splinecode"
                    style={{ width: '100%', height: '100%' }}
                  />
                  {/* Solid rectangle to cover "Built with Spline" logo */}
                  <div 
                    className="absolute bottom-0 right-0 w-64 h-16 z-[9999]"
                    style={{ pointerEvents: 'none', backgroundColor: '#f9f8ff' }}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};