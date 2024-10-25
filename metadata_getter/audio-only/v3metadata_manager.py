import json
from pathlib import Path
from colorama import init, Fore, Style
import inquirer
from datetime import datetime
import logging
import os

# Initialize colorama
init()

class MetadataManager:
    def __init__(self):
        self.current_dir = Path.cwd()
        self.setup_logging()

    def setup_logging(self):
        log_dir = self.current_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'metadata_manager.log'),
                logging.StreamHandler()
            ]
        )

    def select_directory(self):
        """Allow user to select a directory"""
        current_path = self.current_dir
        while True:
            print(f"\n{Fore.CYAN}Current directory: {current_path}{Style.RESET_ALL}")
            directories = [d for d in current_path.iterdir() if d.is_dir()]
            
            choices = []
            if current_path != self.current_dir:
                choices.append(('.. (Go back)', '..'))
            choices.extend([(f"📁 {d.name}", str(d)) for d in directories])
            choices.append(('✓ Select this directory', str(current_path)))

            questions = [
                inquirer.List('path',
                            message="Select directory or navigate",
                            choices=[c for c in choices])
            ]

            answer = inquirer.prompt(questions)
            if not answer:
                return None

            selected = Path(answer['path'])
            if selected.name == '..':
                current_path = current_path.parent
            elif str(selected) == str(current_path):
                return current_path
            else:
                current_path = selected

    def list_wav_files(self, directory):
        """List all WAV files in specified directory"""
        return list(directory.glob("*.wav"))

    def select_wav_file(self, wav_files):
        """Prompt user to select a WAV file"""
        if not wav_files:
            print(f"{Fore.RED}No WAV files found in selected directory!{Style.RESET_ALL}")
            return None

        questions = [
            inquirer.List('wav_file',
                         message="Select WAV file to add metadata",
                         choices=[f.name for f in wav_files])
        ]
        
        answers = inquirer.prompt(questions)
        return answers['wav_file'] if answers else None

    def get_speaker_count(self):
        """Prompt user for number of speakers"""
        while True:
            try:
                count = input(f"\n{Fore.GREEN}Enter number of speakers (1-10): {Style.RESET_ALL}")
                count = int(count)
                if 1 <= count <= 10:
                    return count
                print(f"{Fore.RED}Please enter a number between 1 and 10{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")

    def save_metadata(self, directory, wav_filename, speaker_count):
        """Save metadata in WhisperX compatible format"""
        wav_file = Path(directory) / wav_filename
        logging.info(f"Processing metadata for {wav_file}")

        # Extract event details from filename
        filename_parts = wav_file.stem.split('_')
        if len(filename_parts) >= 6:  # audio_only_Event_Name_YYYYMMDD_HHMMSS
            date_str = filename_parts[-2]  # YYYYMMDD format
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            event_name = '_'.join(filename_parts[2:-2])
        else:
            event_name = wav_file.stem
            formatted_date = datetime.now().strftime("%Y-%m-%d")

        # Create metadata with attendees list for WhisperX compatibility
        metadata = {
            'title': event_name,
            'speaker_count': speaker_count,
            'date': formatted_date,
            'attendees': [{'name': f'Speaker_{i+1}'} for i in range(speaker_count)],
            'file_name': wav_filename,
            'date_modified': datetime.now().isoformat(),
            'metadata_version': '1.0'
        }

        # Save with _metadata.json suffix to match WhisperX expectations
        metadata_file = wav_file.parent / f"{wav_file.stem}_metadata.json"
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            logging.info(f"Metadata saved to {metadata_file}")
            print(f"\n{Fore.GREEN}✓ Metadata saved successfully to: {metadata_file.name}{Style.RESET_ALL}")
            return True
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")
            print(f"\n{Fore.RED}Error saving metadata: {e}{Style.RESET_ALL}")
            return False

    def verify_metadata(self, directory, wav_filename):
        """Verify metadata in WhisperX format"""
        wav_file = Path(directory) / wav_filename
        print(f"\n{Fore.CYAN}=== Verifying Metadata ==={Style.RESET_ALL}")
        print(f"File: {wav_file.name}")

        metadata_file = wav_file.parent / f"{wav_file.stem}_metadata.json"
        logging.info(f"Verifying metadata file: {metadata_file}")

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"\n{Fore.GREEN}✓ Found metadata file: {metadata_file.name}{Style.RESET_ALL}")
                print("\nMetadata contents:")
                for key, value in metadata.items():
                    print(f"  • {key}: {value}")

                # Verify required fields for WhisperX
                required_fields = ['speaker_count', 'title', 'date', 'attendees']
                missing_fields = [field for field in required_fields if field not in metadata]

                if missing_fields:
                    print(f"\n{Fore.YELLOW}⚠ Missing required fields: {', '.join(missing_fields)}{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.GREEN}✓ All required fields present{Style.RESET_ALL}")
                    print(f"Speaker count: {metadata['speaker_count']}")
                    if 'attendees' in metadata:
                        print("Attendees:")
                        for attendee in metadata['attendees']:
                            print(f"  • {attendee['name']}")

                return True

            except Exception as e:
                logging.error(f"Error reading metadata: {e}")
                print(f"{Fore.RED}Error reading metadata file: {e}{Style.RESET_ALL}")
                return False
        else:
            logging.warning(f"No metadata file found: {metadata_file}")
            print(f"{Fore.RED}✗ No metadata file found ({metadata_file.name}){Style.RESET_ALL}")
            return False

    def run(self):
        print(f"\n{Fore.CYAN}=== WAV File Metadata Manager ==={Style.RESET_ALL}")

        try:
            # Select directory
            print("\nSelect directory containing WAV files:")
            selected_dir = self.select_directory()
            if not selected_dir:
                return

            # List WAV files
            wav_files = self.list_wav_files(selected_dir)
            if not wav_files:
                print(f"\n{Fore.RED}No WAV files found in {selected_dir}{Style.RESET_ALL}")
                return

            # Select WAV file
            wav_filename = self.select_wav_file(wav_files)
            if not wav_filename:
                return

            # Menu for actions
            questions = [
                inquirer.List('action',
                             message="Select action",
                             choices=['Add/Update Metadata', 'Verify Metadata', 'Both'])
            ]

            answers = inquirer.prompt(questions)
            action = answers['action']

            if action in ['Add/Update Metadata', 'Both']:
                speaker_count = self.get_speaker_count()
                self.save_metadata(selected_dir, wav_filename, speaker_count)

            if action in ['Verify Metadata', 'Both']:
                self.verify_metadata(selected_dir, wav_filename)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"\n{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")

def main():
    manager = MetadataManager()
    manager.run()

if __name__ == "__main__":
    main()

