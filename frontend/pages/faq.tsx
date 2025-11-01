import React from 'react';
import { Navbar } from '@/components/Navbar';

const FAQPage = () => {
  const faqs = [
    {
      question: "What is TeeTee?",
      answer: "TeeTee is a decentralized AI inference platform that uses Trusted Execution Environments (TEE) to provide secure, verifiable AI computations."
    },
    {
      question: "How do I get started?",
      answer: "Connect your wallet, purchase tokens or get an INFT, and start chatting with our AI models."
    },
    {
      question: "What is an INFT?",
      answer: "INFT (Inference NFT) is a special token that gives you free access to AI inference without consuming tokens."
    },
    {
      question: "How does 0G Storage work?",
      answer: "Your chat conversations are automatically saved to 0G decentralized storage, ensuring data persistence and ownership."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-white to-purple-50">
      <Navbar />
      
      <main className="pt-32 pb-16 px-6">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text">
            Frequently Asked Questions
          </h1>
          <p className="text-gray-600 mb-12">
            Find answers to common questions about TeeTee
          </p>

          <div className="space-y-6">
            {faqs.map((faq, index) => (
              <div 
                key={index}
                className="bg-white rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow"
              >
                <h2 className="text-xl font-semibold text-gray-900 mb-3">
                  {faq.question}
                </h2>
                <p className="text-gray-600 leading-relaxed">
                  {faq.answer}
                </p>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default FAQPage;

