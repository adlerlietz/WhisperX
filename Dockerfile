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

# Download Whisper models ahead of time for faster startup
RUN python3 -c "import whisper; whisper.load_model('base')"
RUN python3 -c "import whisper; whisper.load_model('small')"
RUN python3 -c "import whisper; whisper.load_model('medium')"
RUN python3 -c "import whisper; whisper.load_model('large-v2')"

# Create working directory
WORKDIR /app

# Copy the handler script
COPY handler.py /app/handler.py

# RunPod expects this
CMD ["python3", "-u", "handler.py"] 