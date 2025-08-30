"use client";

import { Suspense, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import ErrorBoundary from "./ErrorBoundary";

// Dynamically import Spline component with no SSR
const Spline = dynamic(() => import("@splinetool/react-spline"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[600px] flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600 mx-auto mb-4"></div>
        <p className="font-mono text-gray-600">Loading 3D Component...</p>
      </div>
    </div>
  )
});

export default function Keyboard3D({ onKeyPress }: { onKeyPress?: (key: number) => void }) {
  const [isMounted, setIsMounted] = useState(false);
  const [hasError, setHasError] = useState(false);
  
  // Only render on client side
  useEffect(() => {
    try {
      setIsMounted(true);
    } catch (error) {
      console.error("Error mounting Keyboard3D component:", error);
      setHasError(true);
    }
  }, []);
  const handleKeyPress = (e: any) => {
    try {
      // Example: detect which object is clicked in Spline
      const objectName = e.target?.name; // Name of the pressed object in Spline

      if (objectName && objectName.startsWith("Key")) {
        const keyNumber = parseInt(objectName.replace("Key", ""), 10);
        if (keyNumber >= 1 && keyNumber <= 5) {
          onKeyPress?.(keyNumber);
        }
      }
    } catch (error) {
      console.log("Error handling key press:", error);
    }
  };

  useEffect(() => {
    // Remove Spline watermark after component mounts
    const removeWatermark = () => {
      const watermarkSelectors = [
        '.spline-watermark',
        '[data-spline-watermark]',
        '.spline-logo',
        'a[href*="spline.design"]'
      ];
      
      watermarkSelectors.forEach(selector => {
        const elements = document.querySelectorAll(selector);
        elements.forEach(el => {
          (el as HTMLElement).style.display = 'none';
        });
      });
    };

    // Run removal immediately and after a delay to catch dynamic elements
    removeWatermark();
    const interval = setInterval(removeWatermark, 1000);
    
    return () => clearInterval(interval);
  }, []);

  // Only render on client-side to prevent SSR errors
  if (!isMounted) {
    return (
      <div className="w-full h-[600px] flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600 mx-auto mb-4"></div>
          <p className="font-mono text-gray-600">Initializing...</p>
        </div>
      </div>
    );
  }
  
  // Display error UI if something went wrong
  if (hasError) {
    return (
      <div className="w-full h-[600px] flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300">
        <div className="text-center p-6">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-amber-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <p className="font-medium text-gray-700 mb-2">Unable to load 3D component</p>
          <p className="text-sm text-gray-500">Please try refreshing the page</p>
        </div>
      </div>
    );
  }

  const errorFallback = (
    <div className="w-full h-[600px] flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300">
      <div className="text-center p-6">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-amber-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <p className="font-medium text-gray-700 mb-2">Unable to load 3D component</p>
        <p className="text-sm text-gray-500">Please try refreshing the page</p>
      </div>
    </div>
  );

  return (
    <div className="w-full h-[600px] relative overflow-hidden">
      <ErrorBoundary fallback={errorFallback}>
        <Suspense 
          fallback={
            <div className="w-full h-[600px] flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-600 mx-auto mb-4"></div>
                <p className="font-mono text-gray-600">Loading 3D Keyboard...</p>
              </div>
            </div>
          }
        >
          <Spline
            scene="https://prod.spline.design/wprODG-IfUxI6A3K/scene.splinecode"
            onMouseDown={handleKeyPress}
          />
        </Suspense>
      </ErrorBoundary>
    </div>
  );
}
