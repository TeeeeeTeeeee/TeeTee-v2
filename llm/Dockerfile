# Use Node.js LTS version
FROM node:20-alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm i 

# Copy application files
COPY . .

# Expose port
EXPOSE 3001

# Environment variable for port (can be overridden)
ENV PORT=3001

# Health check using environment variable
# This will work in any environment (localhost, Phala TEE, cloud, etc.)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the server
CMD ["npm", "start"]

