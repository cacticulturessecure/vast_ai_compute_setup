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
from typing import Dict, List, Optional, Tuple
import time

# Initialize colorama
init()

class WhisperXProcessor:
    def __init__(self):
        # Base directories
        self.workspace_dir = Path("/workspace/audio")
        self.output_base_dir = self.workspace_dir / "audio-only"
        
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        self.hf_token = "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"

        # Disable TF32 for consistency
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logging.info("TF32 disabled for consistency")

    def setup_logging(self):
        """Configure logging system"""
        log_dir = self.workspace_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'whisperx_processing.log'),
                logging.StreamHandler()
            ]
        )

    def load_metadata(self, wav_file: Path) -> Tuple[Optional[Dict], str]:
        """
        Load metadata for the audio file and return both metadata and its location
        """
        try:
            # Define all possible metadata locations
            metadata_locations = [
                (wav_file.parent / f"{wav_file.stem}_metadata.json", "parent directory"),
                (self.workspace_dir / f"{wav_file.stem}_metadata.json", "workspace root"),
                (self.output_base_dir / wav_file.stem / f"{wav_file.stem}_metadata.json", "output directory")
            ]

            logging.info(f"Searching for metadata for {wav_file.name} in:")
            for location, desc in metadata_locations:
                logging.info(f"- {location} ({desc})")
                if location.exists():
                    with open(location, 'r') as f:
                        metadata = json.load(f)
                        if 'speaker_count' in metadata:
                            logging.info(f"Found metadata at {location} with {metadata['speaker_count']} speakers")
                            print(f"{Fore.GREEN}✓ Found metadata file: {location}{Style.RESET_ALL}")
                            return metadata, str(location)

            logging.warning(f"No valid metadata found for {wav_file.name}")
            return None, ""

        except Exception as e:
            logging.error(f"Error loading metadata for {wav_file}: {e}", exc_info=True)
            return None, ""

    def process_audio_file(self, audio_path: Path, output_dir: Path):
        """Process a single audio file with WhisperX"""
        print(f"\n{Fore.CYAN}=== Processing Audio File ==={Style.RESET_ALL}")
        print(f"🎤 File: {audio_path.name}")
        print(f"📂 Output directory: {output_dir}")

        # Load metadata and get its location
        metadata, metadata_location = self.load_metadata(audio_path)
        
        if metadata and 'speaker_count' in metadata:
            speaker_count = metadata['speaker_count']
            print(f"\n{Fore.GREEN}✓ Using metadata from: {metadata_location}")
            print(f"👥 Number of speakers: {speaker_count}")
            if 'attendees' in metadata:
                print("Attendees:")
                for attendee in metadata['attendees']:
                    print(f"  • {attendee['name']}")
        else:
            speaker_count = 2  # Default fallback
            print(f"\n{Fore.YELLOW}⚠ No metadata found, defaulting to {speaker_count} speakers{Style.RESET_ALL}")

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
            self.save_results(result, audio_path, output_dir, metadata)

            print(f"\n{Fore.GREEN}✓ Processing completed successfully!{Style.RESET_ALL}")
            return True

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {e}", exc_info=True)
            print(f"\n{Fore.RED}❌ Error processing file: {str(e)}{Style.RESET_ALL}")
            return False

    def save_results(self, result: Dict, audio_path: Path, output_dir: Path, metadata: Optional[Dict]):
        """Save processing results with speaker mapping"""
        base_name = audio_path.stem
        
        # Create speaker mapping if metadata exists
        speaker_mapping = {}
        if metadata and 'attendees' in metadata:
            for i, attendee in enumerate(metadata['attendees'], 1):
                speaker_mapping[f'SPEAKER_{i}'] = attendee['name']

        # Save detailed transcript with speaker mapping
        transcript_path = output_dir / f"{base_name}.json"
        self.save_transcript(result, transcript_path, speaker_mapping)

        # Save conversation format
        conversation_path = output_dir / f"{base_name}_conversation.json"
        self.save_conversation(result, conversation_path, speaker_mapping)

        # Save text format
        text_path = output_dir / f"{base_name}.txt"
        self.save_text_format(result, text_path, speaker_mapping)

        print(f"\n{Fore.GREEN}Results saved:")
        print(f"  📝 Transcript: {transcript_path}")
        print(f"  💭 Conversation: {conversation_path}")
        print(f"  📄 Text: {text_path}{Style.RESET_ALL}")

    def save_transcript(self, result: Dict, output_file: Path, speaker_mapping: Dict):
        """Save detailed transcript as JSON with mapped speaker names"""
        segments = result["segments"]
        for segment in segments:
            if 'speaker' in segment:
                segment['speaker'] = speaker_mapping.get(segment['speaker'], segment['speaker'])

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved detailed transcript to {output_file}")

    def save_conversation(self, result: Dict, output_file: Path, speaker_mapping: Dict):
        """Save conversation format as JSON with mapped speaker names"""
        conversation = []
        current_speaker = None
        current_text = ''

        for segment in result["segments"]:
            speaker = segment.get('speaker', 'Unknown')
            speaker = speaker_mapping.get(speaker, speaker)
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

        if current_speaker is not None and current_text:
            conversation.append({
                'speaker': current_speaker,
                'text': current_text.strip()
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved conversation format to {output_file}")

    def save_text_format(self, result: Dict, output_file: Path, speaker_mapping: Dict):
        """Save transcript in readable text format with mapped speaker names"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in result["segments"]:
                speaker = segment.get('speaker', 'Unknown')
                speaker = speaker_mapping.get(speaker, speaker)
                text = segment['text'].strip()
                f.write(f"{speaker}: {text}\n")
        logging.info(f"Saved text format to {output_file}")

    def process_all_files(self):
        """Process all WAV files in the workspace directory"""
        print(f"\n{Fore.CYAN}=== WhisperX Audio Processing ==={Style.RESET_ALL}")
        print(f"📂 Scanning directory: {self.workspace_dir}")

        # Debug directory contents
        logging.info(f"Workspace directory contents: {list(self.workspace_dir.glob('*'))}")
        
        # Look for WAV files
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

            try:
                # Create output directory based on the WAV filename
                filename_parts = wav_path.stem.split('_')
                if len(filename_parts) >= 6:  # audio_only_Event_Name_YYYYMMDD_HHMMSS
                    event_parts = filename_parts[2:-2]
                    event_name = '_'.join(event_parts)
                    date_str = filename_parts[-2]
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    dir_name = f"{event_name}_{formatted_date}"
                else:
                    dir_name = wav_path.stem

                output_dir = self.output_base_dir / dir_name
                output_dir.mkdir(parents=True, exist_ok=True)

                if self.process_audio_file(wav_path, output_dir):
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logging.error(f"Error processing {wav_path}: {e}")
                failed += 1
                continue

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
