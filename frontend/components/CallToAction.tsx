import { useState } from "react";
import CountUp from "./CountUp";

export const CallToAction = () => {
  return (
    <section className="py-20 text-center">
        <h2 className="text-[72px] font-bold mb-4">
          Ready to Build with{' '}
          <span className="text-violet-400">TeeTee?</span>
        </h2>
        <p className="text-2xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
          Join thousands of developers building the future of decentralized AI. 
          Get started in minutes or contribute a node to earn rewards.
        </p>
        <div className="flex justify-center gap-6 mb-14">
          <button className="bg-violet-400 text-white px-10 py-4 rounded-full text-xl font-semibold hover:bg-violet-500 transition-colors">
            Get Started Now
          </button>
          <button className="border-2 border-violet-400 text-violet-400 px-10 py-4 rounded-full text-xl font-semibold hover:bg-violet-50 transition-colors">
            Contribute a Node
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-16 max-w-5xl mx-auto">
          <div>
            <p className="text-5xl font-bold text-violet-400 mb-3">
              <CountUp 
                from={0} 
                to={1247} 
                duration={2} 
                separator="," 
                className="text-5xl font-bold text-violet-400"
              />
            </p>
            <p className="text-xl text-gray-600">Active Nodes</p>
          </div>
          <div>
            <p className="text-5xl font-bold text-violet-400 mb-3">
              <CountUp 
                from={0} 
                to={50} 
                duration={1.5} 
                className="text-5xl font-bold text-violet-400"
              />
              <span>M+</span>
            </p>
            <p className="text-xl text-gray-600">Inferences</p>
          </div>
          <div>
            <p className="text-5xl font-bold text-violet-400 mb-3">
              <CountUp 
                from={0} 
                to={15} 
                duration={1.7} 
                className="text-5xl font-bold text-violet-400"
              />
              <span>K+</span>
            </p>
            <p className="text-xl text-gray-600">Developers</p>
          </div>
          <div>
            <p className="text-5xl font-bold text-violet-400 mb-3">
              <CountUp 
                from={0} 
                to={99.8} 
                duration={2.2} 
                className="text-5xl font-bold text-violet-400"
              />
              <span>%</span>
            </p>
            <p className="text-xl text-gray-600">Uptime</p>
          </div>
        </div>
      </section>
  );
};