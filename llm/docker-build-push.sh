#!/bin/bash

# Docker Build and Push Script for AMD64
# Usage: ./docker-build-push.sh YOUR_DOCKERHUB_USERNAME

set -e

# Check if username is provided
if [ -z "$1" ]; then
    echo "Usage: ./docker-build-push.sh YOUR_DOCKERHUB_USERNAME"
    echo "Example: ./docker-build-push.sh derek2403"
    exit 1
fi

DOCKERHUB_USERNAME=$1
IMAGE_NAME="teetee-llm-server"
VERSION="v1.0.0"

echo "🐳 Building Docker image for AMD64..."
echo "📦 Image: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}"
echo ""

# Build for AMD64 platform
docker buildx build \
  --platform linux/amd64 \
  -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest \
  -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION} \
  --load \
  .

echo ""
echo "✅ Build complete!"
echo ""
echo "🔐 Logging in to Docker Hub..."

# Login to Docker Hub
docker login

echo ""
echo "📤 Pushing to Docker Hub..."
echo ""

# Push both tags
docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest
docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}

echo ""
echo "🎉 Successfully pushed to Docker Hub!"
echo ""
echo "Your image is now available at:"
echo "  docker pull ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest"
echo "  docker pull ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""
echo "To run it:"
echo "  docker run -d -p 3001:3001 -e PHALA_API_KEY=your_key ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:latest"
echo ""

