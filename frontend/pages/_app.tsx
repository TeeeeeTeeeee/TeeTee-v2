import "@/styles/globals.css";
import type { AppProps } from "next/app";
import { Pacifico } from "next/font/google";
import { Inter } from "next/font/google";

import Providers from '@/lib/provider';

const pacifico = Pacifico({
  subsets: ["latin"],
  weight: "400", // Pacifico only has regular weight
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export default function App({ Component, pageProps }: AppProps) {
  return (
    <Providers>
      <main className={`${inter.variable} font-sans`}>
        <style jsx global>{`
          :root {
            --font-pacifico: ${pacifico.style.fontFamily};
          }
        `}</style>
        <Component {...pageProps} />
      </main>
    </Providers>
  );
}
