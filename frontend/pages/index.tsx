import type { NextPage } from 'next';
import { Geist, Geist_Mono } from "next/font/google";
import dynamic from 'next/dynamic';
import { Hero, Features, Footer } from '../components';

// Dynamically import components that use framer-motion to avoid SSR issues
const HowItWorks = dynamic(() => import('../components/HowItWorks').then(mod => ({ default: mod.HowItWorks })), { ssr: false });
const CallToAction = dynamic(() => import('../components/CallToAction').then(mod => ({ default: mod.CallToAction })), { ssr: false });

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono", 
  subsets: ["latin"],
});

const Home: NextPage = () => {
  return (
    <main className={`min-h-screen w-full bg-gradient-to-l from-violet-200/20 to-white ${geistSans.variable} ${geistMono.variable}`}>
      <div className="relative space-y-0">
        <Hero />
        <HowItWorks />
        <Features />
        <CallToAction />
        <Footer />
      </div>
    </main>
  );
};

export default Home;