#!/usr/bin/env python3
"""
RunPod Handler for WhisperX with Diarization
"""

import runpod
import whisperx
import os
import gc
import torch
import tempfile
import requests
from urllib.parse import urlparse

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# Cache for models to avoid reloading
model_cache = {}
diarize_model = None

def download_audio(url):
    """Download audio from URL to temporary file"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Create temporary file
        suffix = os.path.splitext(urlparse(url).path)[1] or '.wav'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # Write content
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise Exception(f"Failed to download audio: {str(e)}")

def load_model(model_name="large-v2"):
    """Load and cache WhisperX model"""
    global model_cache
    
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = whisperx.load_model(
            model_name, 
            device, 
            compute_type=compute_type,
            language="en"  # Can be made dynamic
        )
    
    return model_cache[model_name]

def load_diarization_pipeline(hf_token=None):
    """Load and cache diarization pipeline"""
    global diarize_model
    
    if diarize_model is None and hf_token:
        print("Loading diarization pipeline...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
    
    return diarize_model

def handler(job):
    """
    RunPod handler function
    Expected input format:
    {
        "audio": "URL or base64",
        "model": "large-v2",
        "diarize": true,
        "min_speakers": 2,
        "max_speakers": 5,
        "hf_token": "your_token",
        "language": "en",
        "batch_size": 16
    }
    """
    try:
        job_input = job["input"]
        
        # Get parameters
        audio_input = job_input.get("audio")
        model_name = job_input.get("model", "large-v2")
        should_diarize = job_input.get("diarize", True)
        min_speakers = job_input.get("min_speakers", None)
        max_speakers = job_input.get("max_speakers", None)
        hf_token = job_input.get("hf_token") or os.getenv("HF_TOKEN")
        language = job_input.get("language", None)
        batch_size = job_input.get("batch_size", 16)
        
        # Handle audio input
        audio_file = None
        if audio_input.startswith(('http://', 'https://')):
            print(f"Downloading audio from URL: {audio_input}")
            audio_file = download_audio(audio_input)
        else:
            # Handle base64 if needed
            raise Exception("Base64 audio not implemented yet")
        
        # Load model
        model = load_model(model_name)
        
        # Load audio
        print("Loading audio file...")
        audio = whisperx.load_audio(audio_file)
        
        # Transcribe
        print("Transcribing...")
        result = model.transcribe(
            audio, 
            batch_size=batch_size,
            language=language
        )
        
        # Align whisper output
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
        
        # Diarization if requested
        if should_diarize and hf_token:
            print("Running diarization...")
            diarize_pipeline = load_diarization_pipeline(hf_token)
            
            if diarize_pipeline:
                diarize_segments = diarize_pipeline(
                    audio_file,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                
                # Assign speakers
                result = whisperx.assign_word_speakers(
                    diarize_segments, 
                    result
                )
                
                print(f"Diarization complete. Found {len(set(s.get('speaker', 'UNKNOWN') for s in result['segments']))} speakers")
        
        # Format output
        output = {
            "segments": result.get("segments", []),
            "language": result.get("language", language),
            "duration": len(audio) / 16000,  # Assuming 16kHz
            "transcription": " ".join([s["text"] for s in result.get("segments", [])])
        }
        
        # Add speaker count if diarized
        if should_diarize:
            speakers = set(s.get("speaker", "UNKNOWN") for s in output["segments"])
            output["speakers"] = list(speakers)
            output["num_speakers"] = len(speakers)
        
        # Cleanup
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return output
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if 'audio_file' in locals() and audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        
        raise e

# RunPod serverless entrypoint
runpod.serverless.start({
    "handler": handler
}) 