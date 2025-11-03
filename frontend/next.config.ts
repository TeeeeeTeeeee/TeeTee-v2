import type { NextConfig } from "next";

const nextConfig = {
  /* config options here */
  reactStrictMode: true,
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  // Turbopack configuration - no aliases needed, let Spline work normally
  turbopack: {
    // Empty for now - can add custom loaders or aliases here if needed
  },
  devIndicators: false,
} as NextConfig;

export default nextConfig;
