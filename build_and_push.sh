#!/bin/bash

# Replace 'yourusername' with your Docker Hub username
DOCKER_USERNAME="yourusername"
IMAGE_NAME="whisperx-diarization"
TAG="latest"

echo "🐳 Building WhisperX image..."

# Use the simpler Dockerfile
docker build -f Dockerfile.simple -t $DOCKER_USERNAME/$IMAGE_NAME:$TAG .

echo "📤 Pushing to Docker Hub..."
echo "Make sure you're logged in: docker login"

docker push $DOCKER_USERNAME/$IMAGE_NAME:$TAG

echo "✅ Done! Use this in RunPod: $DOCKER_USERNAME/$IMAGE_NAME:$TAG" 