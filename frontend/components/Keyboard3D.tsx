"use client";

import { Suspense, useEffect } from "react";
import Spline from "@splinetool/react-spline";

export default function Keyboard3D({ onKeyPress }: { onKeyPress?: (key: number) => void }) {
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

  return (
    <div className="w-full h-[600px] relative overflow-hidden">
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
    </div>
  );
}
