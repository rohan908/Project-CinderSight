/** @type {import('next').NextConfig} */
const nextConfig = {
  // No rewrites needed since we'll call Railway backend directly
  // The frontend will use NEXT_PUBLIC_API_URL to make direct calls
};

module.exports = nextConfig; 