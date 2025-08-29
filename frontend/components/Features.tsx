interface Feature {
  title: string;
  description: string;
  icon: React.ReactNode;
  gradient: string;
}

const features: Feature[] = [
  {
    title: "Verifiable Computation",
    description: "Every inference is cryptographically proven and verifiable on-chain through TEE attestations.",
    icon: "✓",
    gradient: "bg-gradient-to-r from-violet-400 to-violet-300"
  },
  {
    title: "Model Sharding",
    description: "Large language models are automatically partitioned across multiple TEE nodes for scalability.",
    icon: "⚡",
    gradient: "bg-gradient-to-r from-violet-300 to-purple-300"
  },
  {
    title: "Decentralized Network",
    description: "Distributed infrastructure eliminates single points of failure and censorship resistance.",
    icon: "🌐",
    gradient: "bg-gradient-to-r from-purple-300 to-violet-400"
  },
  {
    title: "Lightning Fast",
    description: "Optimized routing and parallel processing deliver sub-second response times.",
    icon: "⚡",
    gradient: "bg-gradient-to-r from-violet-400 to-purple-300"
  },
  {
    title: "Privacy Preserving",
    description: "TEE isolation ensures data privacy while maintaining computational integrity.",
    icon: "🔒",
    gradient: "bg-gradient-to-r from-purple-300 to-violet-300"
  },
  {
    title: "Cost Efficient",
    description: "Competitive pricing through decentralized resource allocation and optimization.",
    icon: "💰",
    gradient: "bg-gradient-to-r from-violet-300 to-violet-400"
  }
];

export const Features = () => {
  return (
      <section className="py-32 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-[72px] font-bold text-center mb-6">Key Features</h2>
          <p className="text-2xl text-gray-600 text-center mb-20 max-w-4xl mx-auto">
            Discover the revolutionary capabilities that make TeeTee the future of AI inference
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, i) => (
              <div key={i} className="bg-white/90 p-8 rounded-2xl border border-[#DAD9F4]/30">
                <div className={`w-16 h-16 rounded-2xl mb-8 flex items-center justify-center text-white text-2xl ${feature.gradient}`}>
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-bold text-gray-800 mb-4">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
                <a href="#" className="text-[#9A92D9] font-medium mt-6 inline-block">Learn More</a>
              </div>
            ))}
          </div>
        </div>
      </section>
  );
};