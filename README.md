# WhisperX with Diarization on RunPod

This repository contains everything you need to deploy WhisperX with real speaker diarization on RunPod.

## 🚀 Quick Start

### Option 1: Deploy as Serverless Endpoint

1. **Build and push the Docker image:**
```bash
# Replace 'yourusername' with your Docker Hub username
docker build -t yourusername/whisperx-diarization .
docker push yourusername/whisperx-diarization
```

2. **Create RunPod Serverless Endpoint:**
   - Go to [RunPod Console](https://runpod.io/console/serverless)
   - Click "New Endpoint"
   - Container Image: `yourusername/whisperx-diarization:latest`
   - Select GPU: A10 or better
   - Add Environment Variable:
     - `HF_TOKEN`: Your HuggingFace token from [here](https://huggingface.co/settings/tokens)

3. **Update your CallScript.io configuration:**
```python
# In your code, update the endpoint URL
WHISPERX_ENDPOINT = "https://api.runpod.ai/v2/YOUR_NEW_ENDPOINT_ID/run"
```

### Option 2: Use Startup Script (Faster Testing)

If you want to test quickly without building a Docker image, use this startup script:

1. **Create a RunPod GPU Pod** (not serverless)
2. **Use this startup script:**

```bash
#!/bin/bash
# startup.sh - Install WhisperX with diarization on pod boot

# Update system
apt-get update && apt-get install -y ffmpeg git python3-pip

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX
pip install git+https://github.com/m-bain/whisperx.git

# Install Pyannote
pip install pyannote.audio transformers accelerate

# Download models
python3 -c "import whisper; whisper.load_model('large-v2')"

# Create a simple test script
cat > /workspace/test_whisperx.py << 'EOF'
import whisperx
import sys

# Load model
model = whisperx.load_model("large-v2", "cuda", compute_type="float16")

# Transcribe
audio = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
result = model.transcribe(audio)

# Diarize (requires HF_TOKEN env var)
import os
if os.getenv("HF_TOKEN"):
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"))
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

# Print results
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    print(f"[{speaker}] {segment['text']}")
EOF

echo "WhisperX ready! Test with: python3 /workspace/test_whisperx.py your_audio.wav"
```

## 📡 API Usage

Once deployed, use the endpoint like this:

```python
import requests
import json

# Your RunPod endpoint
endpoint = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"
headers = {
    "Authorization": "Bearer YOUR_RUNPOD_API_KEY",
    "Content-Type": "application/json"
}

# Request payload
payload = {
    "input": {
        "audio": "https://example.com/audio.wav",
        "model": "large-v2",
        "diarize": True,
        "min_speakers": 2,
        "max_speakers": 5,
        "hf_token": "YOUR_HF_TOKEN",  # Or set as env var
        "language": None,  # Auto-detect
        "batch_size": 16
    }
}

# Submit job
response = requests.post(endpoint, json=payload, headers=headers)
job_id = response.json()["id"]

# Check status
status_url = endpoint.replace("/run", f"/status/{job_id}")
status = requests.get(status_url, headers=headers).json()

# Get results when complete
if status["status"] == "COMPLETED":
    segments = status["output"]["segments"]
    for seg in segments:
        print(f"[{seg['speaker']}] {seg['text']}")
```

## 🔧 Integration with CallScript.io

Update your existing code to use the new endpoint:

```python
# In worker/processors/whisperx_diarization.py
self.endpoint_url = os.getenv("WHISPERX_ENDPOINT", "https://api.runpod.ai/v2/YOUR_NEW_ENDPOINT_ID/run")

# In api/transcription_service_unified.py
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "https://api.runpod.ai/v2/YOUR_NEW_ENDPOINT_ID/run")
```

## 📊 Expected Output Format

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how can I help you?",
      "speaker": "SPEAKER_00",
      "words": [...]
    },
    {
      "start": 2.8,
      "end": 5.2,
      "text": "I'm interested in your services.",
      "speaker": "SPEAKER_01",
      "words": [...]
    }
  ],
  "language": "en",
  "duration": 125.5,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"]
}
```

## 🎯 Performance Tips

1. **GPU Selection**: 
   - A10/A40: Best price/performance for WhisperX
   - A100: For high-volume production

2. **Model Selection**:
   - `base`: Fast, good for short calls
   - `small`/`medium`: Balanced
   - `large-v2`: Best accuracy, slower

3. **Batch Size**: 
   - Increase for better GPU utilization
   - Default 16 works well for most cases

## 🐛 Troubleshooting

### No speaker labels in output
- Ensure `HF_TOKEN` is set correctly
- Check that `diarize: true` in request
- Verify audio quality (stereo works better)

### Slow processing
- Pre-download models in Dockerfile
- Use smaller model for testing
- Ensure GPU is being used (`nvidia-smi`)

### Memory errors
- Reduce batch_size
- Use smaller model
- Process shorter audio segments

## 📝 License

This deployment uses:
- WhisperX: BSD-4-Clause
- Pyannote: MIT
- Whisper: MIT 