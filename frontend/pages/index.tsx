import type { NextPage } from 'next';
import { Geist, Geist_Mono } from "next/font/google";
import { Navbar, Hero, HowItWorks, Features, CallToAction, Footer } from '../components';

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
    <main className={`min-h-screen w-full ${geistSans.variable} ${geistMono.variable}`}>
      <div className="relative">
        <Navbar />
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