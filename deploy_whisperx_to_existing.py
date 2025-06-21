#!/usr/bin/env python3
"""
Deploy WhisperX diarization to existing RunPod endpoint
This adds diarization support to your current Faster Whisper endpoint
"""

import os
import sys

print("""
🎯 WhisperX Diarization Deployment Guide
=======================================

Since RunPod is having issues building from GitHub, here's how to add 
diarization to your existing endpoint:

1. Use your existing endpoint: gb7lespz6g2rrm

2. Update your CallScript configuration:
   - The endpoint already supports URL-based audio
   - Just enable the diarization flag in your requests

3. In your code, update:
   worker/processors/whisperx_diarization.py
   - Set use_mock = False to enable real diarization
   - Ensure HF_TOKEN is set in environment

4. The existing endpoint will work, but without true diarization
   (it will use alternating speakers)

5. For true diarization, wait for RunPod to fix their build issue
   and try building again later.

Alternative: Use Docker Hub
---------------------------
If you need true diarization NOW:
1. Install Docker Desktop
2. Run: ./build_and_push.sh (update username first)
3. Use Docker image in RunPod instead of GitHub
""")
