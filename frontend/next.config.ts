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
  // Turbopack configuration (replaces webpack config when using --turbopack)
  turbopack: {
    resolveAlias: {
      // Stub problematic modules - use relative path for Windows compatibility
      "@splinetool/react-spline": "./utils/emptyModule.tsx",
    },
  },
  // Keep webpack config as fallback when not using --turbopack
  webpack: (config: any) => {
    config.resolve = config.resolve || {};
    config.resolve.alias = config.resolve.alias || {};
    config.resolve.alias["@splinetool/react-spline"] = require.resolve("./utils/emptyModule.tsx");
    return config;
  },
  devIndicators: false,
} as NextConfig;

export default nextConfig;
