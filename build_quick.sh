#!/bin/bash

# Quick WhisperX build for Docker Hub
echo "1. First, create a Docker Hub account at https://hub.docker.com"
echo "2. Replace 'yourusername' below with your Docker Hub username"
echo ""
read -p "Enter your Docker Hub username: " DOCKER_USER

# Build the image
echo "Building WhisperX image..."
docker build -f Dockerfile.simple -t $DOCKER_USER/whisperx:latest .

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login

# Push the image
echo "Pushing image..."
docker push $DOCKER_USER/whisperx:latest

echo "✅ Done! Use this in RunPod: $DOCKER_USER/whisperx:latest"
