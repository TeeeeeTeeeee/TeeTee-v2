"use client";

import { useRef, useEffect, useState } from "react";
import CircularGallery from "./CircularGallery";

interface Feature {
  title: string;
  description: string;
  icon: string;
  gradient: string;
  backgroundColor?: string;
}

const features: Feature[] = [
  {
    title: "Verifiable Computation",
    description: "Every inference is cryptographically proven and verifiable on-chain.",
    icon: "✓",
    gradient: "bg-gradient-to-r from-violet-400 to-violet-300",
    backgroundColor: "#8B5CF6" // Purple-500
  },
  {
    title: "Model Sharding",
    description: "Large language models are automatically partitioned across multiple TEE nodes.",
    icon: "⚡", // Lightning bolt for Model Sharding
    gradient: "bg-gradient-to-r from-violet-300 to-purple-300",
    backgroundColor: "#8B5CF6" // Purple-500 (same as others)
  },
  {
    title: "Decentralized Network",
    description: "Distributed infrastructure eliminates single points of failure.",
    icon: "🌐",
    gradient: "bg-gradient-to-r from-purple-300 to-violet-400",
    backgroundColor: "#8B5CF6" // Purple-500
  },
  {
    title: "Lightning Fast",
    description: "Optimized routing and parallel processing deliver sub-second response times.",
    icon: "speed", // Using the speed/movement symbol option
    gradient: "bg-gradient-to-r from-violet-400 to-purple-300",
    backgroundColor: "#8B5CF6" // Purple-500
  },
  {
    title: "Privacy Preserving",
    description: "TEE isolation ensures data privacy while maintaining integrity.",
    icon: "🔒",
    gradient: "bg-gradient-to-r from-purple-300 to-violet-300",
    backgroundColor: "#8B5CF6" // Purple-500
  },
  {
    title: "Cost Efficient",
    description: "Competitive pricing through decentralized resource allocation.",
    icon: "💰",
    gradient: "bg-gradient-to-r from-violet-300 to-violet-400",
    backgroundColor: "#8B5CF6" // Purple-500
  }
];

export const Features = () => {
  const featuresRef = useRef<HTMLElement>(null);
  const [isMounted, setIsMounted] = useState(false);

  // Set isMounted to true when component mounts on client-side
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Create gallery items with special SVG-like icons that work well with gradient
  const getSvgIcon = (icon: string) => {
    switch(icon) {
      case "✓": return "✓"; // Checkmark
      case "⚡": return "↯"; // Better lightning bolt for Model Sharding with gradient
      case "⚡⚡": return "⤑"; // Arrow symbol for Lightning Fast (Option 1)
      case "fast": return "⟩⟩"; // Double chevron for Lightning Fast (Option 2)
      case "speed": return "⥈"; // Speed/movement symbol for Lightning Fast (Option 3)
      case "🌐": return "⦿"; // Globe symbol
      case "🔒": return "◉"; // Lock symbol
      case "💰": return "⊙"; // Money symbol
      default: return "⬮";
    }
  };
  
  const galleryItems = features.map(feature => ({
    icon: getSvgIcon(feature.icon),
    text: feature.title,
    description: feature.description,
    backgroundColor: feature.backgroundColor // This won't be used since we're using gradient
  }));
  
  return (
    <section ref={featuresRef} className="py-20 bg-white" id="features">
      <div className="max-w-7xl mx-auto px-6">
        <h2 className="text-[72px] font-bold text-center mb-4">Key Features</h2>
        <p className="text-2xl text-gray-600 text-center mb-8 max-w-4xl mx-auto">
          Discover the revolutionary capabilities that make TeeTee the future of AI inference
        </p>
        
        {/* CircularGallery container - only render when component is mounted on client side */}
        <div className="h-[450px] relative mb-0">
          {isMounted && (
            <CircularGallery 
              items={galleryItems}
              bend={2.5}
              textColor="#333333"
              backgroundColor="#8B5CF6" // This won't be used since we're using gradient in the component
              borderRadius={0.05}
              font="bold 16px Figtree"
              scrollSpeed={2}
              scrollEase={0.05}
            />
          )}
        </div>
      </div>
    </section>
  );
};