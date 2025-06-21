FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    wget \
    libsndfile1 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python + pip setup
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install compatible NumPy version (before 2.0)
RUN pip install numpy==1.24.3

# Install WhisperX and dependencies
RUN pip install git+https://github.com/m-bain/whisperx.git

# Install Pyannote for diarization
RUN pip install pyannote.audio==3.1.1 \
    transformers \
    accelerate \
    rich

# Install RunPod SDK and API server dependencies
RUN pip install runpod \
    fastapi \
    uvicorn \
    python-multipart \
    requests

# Models will be downloaded on first use
# Skip pre-loading to avoid build issues

# Create working directory
WORKDIR /app

# Copy the handler script
COPY handler.py /app/handler.py

# RunPod expects this
CMD ["python3", "-u", "handler.py"] 