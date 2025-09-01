/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    // Disable fast refresh to prevent infinite reload loops during development
    forceSwcTransforms: true,
  },
  // Disable fast refresh temporarily
  fastRefresh: false,
};

export default nextConfig;