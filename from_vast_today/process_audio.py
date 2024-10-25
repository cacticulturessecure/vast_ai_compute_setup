import whisperx
import torch
import gc
import os
import json
from pathlib import Path
from datetime import datetime
import logging
from colorama import init, Fore, Style
from tqdm import tqdm
import soundfile as sf
from typing import Dict, List, Optional
import time

# Initialize colorama
init()

class WhisperXProcessor:
    def __init__(self):
        # Base directories
        self.workspace_dir = Path("/workspace/audio")
        self.output_base_dir = self.workspace_dir / "audio-only"
        self.exports_dir = Path("/home/securemeup/workspace/dev-workspace/localdock-to-colab/media/exports")
        
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        self.hf_token = "enter-hugging-face-token"
        
        # Disable TF32 for consistency
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def setup_logging(self):
        """Configure logging system"""
        log_dir = self.workspace_dir / "vast_ai_compute_setup" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'whisperx_processing.log'),
                logging.StreamHandler()
            ]
        )

    def get_output_directory(self, wav_file: Path) -> Path:
        """Create and return the appropriate output directory based on the WAV filename"""
        filename_parts = wav_file.stem.split('_')
        if len(filename_parts) >= 6:  # audio_only_Event_Name_YYYYMMDD_HHMMSS
            # Remove 'audio_only' prefix
            event_parts = filename_parts[2:-2]
            event_name = '_'.join(event_parts)
            date_str = filename_parts[-2]
            
            # Format the date
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            
            # Create directory name
            dir_name = f"{event_name}_{formatted_date}"
            
            # Create and return full path
            output_dir = self.output_base_dir / dir_name
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        else:
            # Fallback: use the filename without extension as directory name
            dir_name = wav_file.stem
            output_dir = self.output_base_dir / dir_name
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir

    def load_metadata(self, wav_file: Path) -> Optional[Dict]:
        """Load metadata for the audio file"""
        try:
            # Extract date from filename
            filename_parts = wav_file.stem.split('_')
            date_str = filename_parts[-2]  # YYYYMMDD format
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            
            # Check exports directory for event data
            exports_date_dir = self.exports_dir / formatted_date
            if exports_date_dir.exists():
                # Look for matching event file
                event_name = '_'.join(filename_parts[2:-2])  # Extract event name from filename
                for json_file in exports_date_dir.glob("*.json"):
                    if json_file.name.endswith('_metadata.json'):
                        continue
                    
                    with open(json_file, 'r') as f:
                        event_data = json.load(f)
                        if 'attendees' in event_data:
                            metadata = {
                                'speaker_count': len(event_data['attendees']),
                                'event_title': event_data.get('title', ''),
                                'date': formatted_date,
                                'attendees': event_data['attendees']
                            }
                            logging.info(f"Found metadata with {metadata['speaker_count']} speakers for {wav_file.name}")
                            return metadata
            
            logging.warning(f"No metadata found for {wav_file}")
            return None
            
        except Exception as e:
            logging.error(f"Error loading metadata for {wav_file}: {e}")
            return None

    def process_audio_file(self, audio_path: Path, output_dir: Path):
        """Process a single audio file with WhisperX"""
        print(f"\n{Fore.CYAN}=== Processing Audio File ==={Style.RESET_ALL}")
        print(f"🎤 File: {audio_path.name}")
        print(f"📂 Output directory: {output_dir}")
        
        # Load metadata to get speaker count
        metadata = self.load_metadata(audio_path)
        if metadata and 'speaker_count' in metadata:
            speaker_count = metadata['speaker_count']
            print(f"👥 Speakers from metadata: {speaker_count}")
            if 'attendees' in metadata:
                print("Attendees:")
                for attendee in metadata['attendees']:
                    print(f"  • {attendee.get('name', 'Unknown')}")
        else:
            speaker_count = 2  # Default fallback
            print(f"{Fore.YELLOW}⚠️ No metadata found, defaulting to {speaker_count} speakers{Style.RESET_ALL}")

        try:
            # Load audio
            print(f"\n{Fore.YELLOW}Loading audio...{Style.RESET_ALL}")
            audio = whisperx.load_audio(str(audio_path))

            # Transcribe with Whisper
            print(f"{Fore.YELLOW}Transcribing with Whisper...{Style.RESET_ALL}")
            model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type,
                language='en'
            )
            
            with tqdm(total=100, desc="Transcribing") as pbar:
                result = model.transcribe(
                    audio,
                    batch_size=16,
                    language='en'
                )
                pbar.update(100)

            # Align transcript
            print(f"\n{Fore.YELLOW}Aligning transcript...{Style.RESET_ALL}")
            model_a, metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device
            )
            
            with tqdm(total=100, desc="Aligning") as pbar:
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device
                )
                pbar.update(100)

            # Clear GPU memory
            del model
            del model_a
            gc.collect()
            torch.cuda.empty_cache()

            # Diarize
            print(f"\n{Fore.YELLOW}Diarizing with {speaker_count} speakers...{Style.RESET_ALL}")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )

            with tqdm(total=100, desc="Diarizing") as pbar:
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=speaker_count,
                    max_speakers=speaker_count
                )
                pbar.update(100)

            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Save results
            base_name = audio_path.stem

            # Save detailed transcript
            transcript_path = output_dir / f"{base_name}.json"
            self.save_transcript(result, transcript_path)

            # Save conversation format
            conversation_path = output_dir / f"{base_name}_conversation.json"
            self.save_conversation(result, conversation_path)

            # Save text format
            text_path = output_dir / f"{base_name}.txt"
            self.save_text_format(result, text_path)

            print(f"\n{Fore.GREEN}✓ Processing completed successfully!")
            print(f"  📝 Transcript: {transcript_path}")
            print(f"  💭 Conversation: {conversation_path}")
            print(f"  📄 Text: {text_path}{Style.RESET_ALL}")

            return True

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {e}")
            print(f"\n{Fore.RED}❌ Error processing file: {str(e)}{Style.RESET_ALL}")
            return False

    def save_transcript(self, result: Dict, output_file: Path):
        """Save detailed transcript as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result["segments"], f, ensure_ascii=False, indent=4)
        logging.info(f"Saved detailed transcript to {output_file}")

    def save_conversation(self, result: Dict, output_file: Path):
        """Save conversation format as JSON"""
        conversation = []
        current_speaker = None
        current_text = ''

        for segment in result["segments"]:
            speaker = segment.get('speaker', 'Unknown')
            text = segment['text'].strip()
            
            if speaker != current_speaker:
                if current_speaker is not None:
                    conversation.append({
                        'speaker': current_speaker,
                        'text': current_text.strip()
                    })
                current_speaker = speaker
                current_text = text + ' '
            else:
                current_text += text + ' '
        
        # Add last segment
        if current_speaker is not None and current_text:
            conversation.append({
                'speaker': current_speaker,
                'text': current_text.strip()
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved conversation format to {output_file}")

    def save_text_format(self, result: Dict, output_file: Path):
        """Save transcript in readable text format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in result["segments"]:
                speaker = segment.get('speaker', 'Unknown')
                text = segment['text'].strip()
                f.write(f"{speaker}: {text}\n")
        logging.info(f"Saved text format to {output_file}")

    def process_all_files(self):
        """Process all WAV files in the workspace directory"""
        print(f"\n{Fore.CYAN}=== WhisperX Audio Processing ==={Style.RESET_ALL}")
        print(f"📂 Scanning directory: {self.workspace_dir}")

        # Look for WAV files directly in the workspace/audio directory
        wav_files = list(self.workspace_dir.glob("*.wav"))
        if not wav_files:
            print(f"\n{Fore.YELLOW}No WAV files found!{Style.RESET_ALL}")
            return

        print(f"\nFound {len(wav_files)} files to process")
        
        successful = 0
        failed = 0

        for i, wav_path in enumerate(wav_files, 1):
            print(f"\n{Fore.CYAN}[File {i}/{len(wav_files)}]{Style.RESET_ALL}")
            print("=" * 50)
            
            # Get the appropriate output directory for this file
            output_dir = self.get_output_directory(wav_path)
            
            # Process the file
            if self.process_audio_file(wav_path, output_dir):
                successful += 1
            else:
                failed += 1

        # Print summary
        print(f"\n{Fore.CYAN}=== Processing Summary ==={Style.RESET_ALL}")
        print(f"✓ Successfully processed: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success rate: {(successful/len(wav_files))*100:.1f}%")

def main():
    try:
        processor = WhisperXProcessor()
        processor.process_all_files()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Processing interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"\n{Fore.RED}An unexpected error occurred. Check logs for details.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
