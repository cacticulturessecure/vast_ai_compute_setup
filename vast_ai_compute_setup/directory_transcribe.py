import whisperx
import torch
import gc
import os
import json
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

def save_transcript(result, output_file):
    """Save detailed transcript as JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=4)
    print(f"Detailed transcript saved to {output_file}")

def save_conversation(result, output_file):
    """Save conversation format as JSON"""
    conversation = []
    current_speaker = None
    current_text = ''

    for segment in result["segments"]:
        speaker = segment.get('speaker', 'Unknown')
        text = segment['text'].strip()
        
        if speaker != current_speaker:
            if current_speaker is not None:
                conversation.append({'speaker': current_speaker, 'text': current_text.strip()})
            current_speaker = speaker
            current_text = text + ' '
        else:
            current_text += text + ' '
    
    # Add last segment
    if current_speaker is not None and current_text:
        conversation.append({'speaker': current_speaker, 'text': current_text.strip()})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)
    print(f"Conversation format saved to {output_file}")

def process_audio_file(audio_path, num_speakers):
    """Process a single audio file with WhisperX"""
    device = "cuda"
    compute_type = "float16"
    
    # Disable TF32 for consistency
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        print(f"Loading audio: {audio_path}")
        audio = whisperx.load_audio(str(audio_path))
        
        print("Transcribing with Whisper (English model)...")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language='en')
        result = model.transcribe(audio, batch_size=16, language='en')
        
        print("Aligning transcript...")
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
        
        # Clear GPU memory
        del model
        del model_a
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Diarizing with {num_speakers} speakers...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
        diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Create output filenames
        base_path = audio_path.with_suffix('')  # Remove .wav extension
        transcript_json_path = base_path.with_suffix('.json')
        conversation_json_path = Path(str(base_path) + '_conversation.json')
        
        print("Saving JSON files...")
        # Save both JSON formats
        save_transcript(result, transcript_json_path)
        save_conversation(result, conversation_json_path)
        
        print(f"✓ Completed! Files saved:\n"
              f"  - Detailed transcript: {transcript_json_path}\n"
              f"  - Conversation format: {conversation_json_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        # Log the error to a file
        with open("transcription_errors.log", "a") as f:
            f.write(f"\nError processing {audio_path}:\n{str(e)}\n")

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
    for i, wav_path in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] Processing file: {wav_path}")
        num_speakers = get_speaker_count(wav_path)
        process_audio_file(wav_path, num_speakers)

if __name__ == "__main__":
    main()
