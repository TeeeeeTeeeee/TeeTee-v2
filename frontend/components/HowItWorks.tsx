import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Step {
  id: number;
  title: string;
  description: string;
  image?: string;
}

const steps: Step[] = [
  {
    id: 1,
    title: "Request Initaition",
    description: "User submits AI inference request through secure API endpoint.",
  },
  {
    id: 2,
    title: "Model Sharding",
    description: "LLM is automatically partitioned across multiple TEE nodes.",
  },
  {
    id: 3,
    title: "Secure Processing",
    description: "Each TEE processes its shard in isolated, verifiable environment.",
  },
  {
    id: 4,
    title: "Attestation",
    description: "Cryptographic proofs generated and verified on-chain.",
  },
  {
    id: 5,
    title: "Result Aggregation",
    description: "Outputs combined and returned with full verifiability.",
  },
];

export const HowItWorks = () => {
  const [currentStep, setCurrentStep] = useState(0);

  // Updated navigation functions with limits
  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  // Add disabled states for buttons
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === steps.length - 1;

  return (
    <section className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        <div className="mb-20">
          <h2 className="font-mono text-[72px] font-bold mb-6">
            How It Works
          </h2>
          <p className="font-mono text-xl text-gray-600 uppercase tracking-wider">
            Experience seamless AI interface through our revolutionary TEE-based architecture
          </p>
        </div>

        <div className="grid grid-cols-2 gap-10">
          {/* Left: Content Display */}
          <div className="bg-white border-2 border-black rounded-none p-12">
            <div className="flex justify-between items-start mb-8">
              {/* Big Step Number */}
              <span className="font-mono text-[120px] leading-none font-bold">
                {String(currentStep + 1).padStart(2, '0')}
              </span>
              {/* Title */}
              <div className="w-1/2 text-right">
                <h3 className="font-mono text-[25px] uppercase">
                  {steps[currentStep].title}
                </h3>
              </div>
            </div>

            {/* Horizontal Line */}
            <div className="border-t-2 border-black my-8"></div>

            {/* Description */}
            <p className="font-mono text-lg leading-relaxed mb-12">
              {steps[currentStep].description}
            </p>

            {/* Navigation */}
            <div className="flex items-center justify-between p-2">
              <button 
                onClick={prevStep}
                disabled={isFirstStep}
                className={`w-40 h-40 border-2 border-black flex items-center justify-center transition-colors ${
                  isFirstStep 
                    ? 'opacity-50 cursor-not-allowed' 
                    : 'hover:bg-black hover:text-white'
                }`}
              >
                <div className="w-20 h-20 flex items-center justify-center transform rotate-180">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 12h16M16 4l8 8-8 8" />
                  </svg>
                </div>
              </button>

              <div className="font-mono">
                {String(currentStep + 1).padStart(2, '0')}/
                {String(steps.length).padStart(2, '0')}
              </div>

              <button 
                onClick={nextStep}
                disabled={isLastStep}
                className={`w-40 h-40 border-2 border-black flex items-center justify-center transition-colors ${
                  isLastStep 
                    ? 'opacity-50 cursor-not-allowed' 
                    : 'hover:bg-black hover:text-white'
                }`}
              >
                <div className="w-20 h-20 flex items-center justify-center">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 12h16M16 4l8 8-8 8" />
                  </svg>
                </div>
              </button>
            </div>
          </div>

          {/* Right: Folder Display */}
          <div className="relative h-[600px] flex items-center justify-center bg-gray-50">
            <AnimatePresence>
              {steps.map((step, index) => {
                const offset = index - currentStep;
                return (
                  <motion.div
                    key={step.id}
                    initial={{ x: 100, opacity: 0 }}
                    animate={{
                      x: offset * 20,
                      y: offset * 10,
                      opacity: offset < 3 && offset >= 0 ? 1 : 0,
                      zIndex: steps.length - Math.abs(offset),
                    }}
                    exit={{ x: -100, opacity: 0 }}
                    transition={{
                      type: "spring",
                      stiffness: 100,
                      damping: 20,
                    }}
                    className="absolute w-full max-w-md bg-white border-2 border-black"
                  >
                    {/* Folder Tab */}
                    <div className="relative">
                      <div className="absolute -top-6 left-8 w-32 h-6 bg-white border-2 border-b-0 border-black flex items-center justify-between px-3">
                        <span className="font-mono text-sm">
                          {String(index + 1).padStart(3, '0')}
                        </span>
                        <span className="font-mono text-sm uppercase">
                          {step.title.slice(0, 1)}
                        </span>
                      </div>
                    </div>

                    {/* Folder Content */}
                    <div className="p-6">
                      <div className="flex justify-between items-start mb-4">
                        <h4 className="font-mono text-xl uppercase">{step.title}</h4>
                        <span className="font-mono text-sm opacity-50">
                          {String(index + 1).padStart(2, '0')}/
                          {String(steps.length).padStart(2, '0')}
                        </span>
                      </div>
                      <div className="h-px bg-black my-4" />
                      {offset === 0 && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="font-mono text-sm leading-relaxed"
                        >
                          <p>{step.description}</p>
                        </motion.div>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
};