import whisperx
import torch
import gc
import os
from pathlib import Path

def get_speaker_count(wav_file):
    """Ask user for speaker count for each WAV file"""
    print(f"\nProcessing: {wav_file}")
    while True:
        try:
            count = int(input(f"How many speakers are in this recording? "))
            if 1 <= count <= 10:  # reasonable limit
                return count
            print("Please enter a reasonable number of speakers (1-10)")
        except ValueError:
            print("Please enter a valid number")

def process_audio_file(audio_path, num_speakers):
    """Process a single audio file with WhisperX"""
    device = "cuda"
    compute_type = "float16"
    
    # Disable TF32 for consistency
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        # 1. Load audio
        audio = whisperx.load_audio(str(audio_path))
        
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
        diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Create transcript filename (same path as wav but with .txt extension)
        transcript_path = audio_path.with_suffix('.txt')
        
        # Save output
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"=== Transcription for {audio_path.name} ===\n")
            f.write(f"Number of speakers: {num_speakers}\n\n")
            for segment in result["segments"]:
                speaker = segment.get("speaker", "Unknown")
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                f.write(f"[{start:.2f}s -> {end:.2f}s] Speaker {speaker}: {text}\n")
        
        print(f"\nTranscription completed! Check {transcript_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")

def main():
    # Base directory containing audio files
    base_dir = Path("/workspace/audio/audio-only")
    
    # Find all WAV files
    wav_files = list(base_dir.rglob("*.wav"))
    
    if not wav_files:
        print("No WAV files found!")
        return
    
    print(f"Found {len(wav_files)} WAV files to process.")
    
    # Process each WAV file
    for wav_path in wav_files:
        print(f"\nProcessing file: {wav_path}")
        num_speakers = get_speaker_count(wav_path)
        process_audio_file(wav_path, num_speakers)

if __name__ == "__main__":
    main()
