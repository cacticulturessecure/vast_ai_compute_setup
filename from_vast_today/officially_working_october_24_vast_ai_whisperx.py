import whisperx
import torch
import gc
import os

# Config
audio_file = "audio.wav"  # Make sure this matches your audio file name
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
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
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
