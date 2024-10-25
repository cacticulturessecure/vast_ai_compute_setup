#!/bin/bash
# WhisperX Setup Guide for vast.ai

# 1. Initial System Updates and Tools
apt-get update
apt-get install -y nano ffmpeg

# 2. Install PyTorch with CUDA support
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install WhisperX and dependencies
pip install git+https://github.com/m-bain/whisperx.git

# 4. Create test script to verify installation
cat <<'EOF' >test_setup.py
import whisperx
import torch

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device:", device)
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name())

# Disable TF32 for consistency
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Load model (this will verify whisperx is working)
try:
    model = whisperx.load_model("large-v2", device, compute_type="float16" if device == "cuda" else "int8")
    print("\nWhisperX model loaded successfully!")
except Exception as e:
    print("\nError loading WhisperX model:", str(e))
EOF

# 5. Create the main transcription script
cat <<'EOF' >transcribe.py
import whisperx
import torch
import gc
import os

# Config
audio_file = "audio.wav"  # Update this to match your audio file name
device = "cuda"
compute_type = "float16"

# Disable TF32 for consistency
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print(f"Processing: {audio_file}")

try:
    # 1. Load audio
    audio = whisperx.load_audio(audio_file)
    
    # 2. Load model and transcribe
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=16)
    
    # 3. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    
    # Clear GPU memory
    del model
    del model_a
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. Diarize with speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="YOUR_HUGGING_FACE_TOKEN", device=device)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Save output
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write("=== Transcription ===\n\n")
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"[{start:.2f}s -> {end:.2f}s] Speaker {speaker}: {text}\n")
    
    print("\nTranscription completed! Check transcript.txt")

except Exception as e:
    print(f"Error occurred: {str(e)}")
EOF

# 6. Make scripts executable
chmod +x test_setup.py transcribe.py

echo "Setup complete! Follow these steps to use:"
echo "1. Run test: python3 test_setup.py"
echo "2. Update Hugging Face token in transcribe.py"
echo "3. Place your audio file in the working directory"
echo "4. Run transcription: python3 transcribe.py"
