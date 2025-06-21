#!/bin/bash
# WhisperX with Diarization - RunPod Startup Script
# This script installs WhisperX and Pyannote on a RunPod GPU pod

echo "🚀 Installing WhisperX with Diarization Support..."

# Update system packages
apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    python3-pip \
    python3-dev \
    build-essential

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX
echo "🎯 Installing WhisperX..."
pip install git+https://github.com/m-bain/whisperx.git

# Install Pyannote and dependencies
echo "🎭 Installing Pyannote for diarization..."
pip install pyannote.audio==3.1.1 \
    transformers \
    accelerate \
    huggingface-hub

# Install API dependencies
echo "🌐 Installing API dependencies..."
pip install fastapi uvicorn requests

# Download Whisper models
echo "📥 Downloading Whisper models..."
python3 -c "import whisper; whisper.load_model('base')"
python3 -c "import whisper; whisper.load_model('large-v2')"

# Create test script
cat > /workspace/test_diarization.py << 'EOF'
#!/usr/bin/env python3
import whisperx
import sys
import os

def transcribe_with_diarization(audio_path, hf_token=None):
    """Test WhisperX with diarization"""
    
    # Device setup
    device = "cuda"
    compute_type = "float16"
    
    # Load model
    print("Loading WhisperX model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    
    # Load audio
    print(f"Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)
    
    # Transcribe
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=16)
    print(f"Detected language: {result['language']}")
    
    # Align
    print("Aligning...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    # Diarize if token provided
    if hf_token:
        print("Running speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        diarize_segments = diarize_model(audio_path)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Count speakers
        speakers = set(s.get("speaker", "UNKNOWN") for s in result["segments"])
        print(f"Found {len(speakers)} speakers: {speakers}")
    
    # Display results
    print("\n=== TRANSCRIPT ===")
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_diarization.py <audio_file> [hf_token]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("⚠️  No HuggingFace token provided - diarization will be skipped")
        print("   Set HF_TOKEN env var or pass as second argument")
    
    transcribe_with_diarization(audio_file, hf_token)
EOF

chmod +x /workspace/test_diarization.py

# Create API server
cat > /workspace/whisperx_server.py << 'EOF'
#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import whisperx
import torch
import os
import tempfile
import requests
from typing import Optional, List

app = FastAPI(title="WhisperX Diarization API")

# Global model cache
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model_cache = {}
diarize_model = None

class TranscriptionRequest(BaseModel):
    audio_url: str
    model: str = "large-v2"
    diarize: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    hf_token: Optional[str] = None
    language: Optional[str] = None

@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    """Transcribe audio with optional diarization"""
    try:
        # Download audio
        response = requests.get(request.audio_url)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(response.content)
            audio_path = f.name
        
        # Load model
        if request.model not in model_cache:
            model_cache[request.model] = whisperx.load_model(
                request.model, device, compute_type=compute_type
            )
        model = model_cache[request.model]
        
        # Transcribe
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, language=request.language)
        
        # Align
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device
        )
        
        # Diarize if requested
        if request.diarize and (request.hf_token or os.getenv("HF_TOKEN")):
            global diarize_model
            if diarize_model is None:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=request.hf_token or os.getenv("HF_TOKEN"),
                    device=device
                )
            
            diarize_segments = diarize_model(
                audio_path,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Cleanup
        os.unlink(audio_path)
        
        return {
            "segments": result["segments"],
            "language": result["language"],
            "duration": len(audio) / 16000
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "device": device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create download script for test audio
cat > /workspace/download_test_audio.sh << 'EOF'
#!/bin/bash
echo "Downloading test audio files..."
wget -O gettysburg.wav https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav
echo "Test audio downloaded: gettysburg.wav"
EOF
chmod +x /workspace/download_test_audio.sh

# Final instructions
echo ""
echo "✅ WhisperX Installation Complete!"
echo ""
echo "📝 Quick Start:"
echo "1. Set your HuggingFace token:"
echo "   export HF_TOKEN='your_token_here'"
echo ""
echo "2. Test transcription with diarization:"
echo "   cd /workspace"
echo "   ./download_test_audio.sh"
echo "   python3 test_diarization.py gettysburg.wav"
echo ""
echo "3. Start API server:"
echo "   python3 whisperx_server.py"
echo ""
echo "🎯 The API will be available at http://localhost:8000"
echo "📚 API docs at http://localhost:8000/docs" 