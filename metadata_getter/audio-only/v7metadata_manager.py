import json
from pathlib import Path
from colorama import init, Fore, Style
import inquirer
from datetime import datetime
import logging
import os
import shutil

# Initialize colorama
init()

class MetadataManager:
    def __init__(self):
        self.current_dir = Path.cwd()
        # Always ensure we have reference to the WhisperX workspace
        self.whisperx_workspace = Path("/workspace/audio")
        self.setup_logging()

    def setup_logging(self):
        """Configure logging with more detailed output"""
        log_dir = self.current_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'metadata_manager.log'),
                logging.StreamHandler()
            ]
        )

    def select_directory(self):
        """Allow user to select a directory with improved navigation"""
        current_path = self.current_dir
        while True:
            print(f"\n{Fore.CYAN}Current directory: {current_path}{Style.RESET_ALL}")
            try:
                directories = sorted([d for d in current_path.iterdir() if d.is_dir()])
                
                choices = []
                if current_path != self.current_dir:
                    choices.append(('.. (Go back)', '..'))
                choices.extend([(f"📁 {d.name}", str(d)) for d in directories])
                choices.append(('✓ Select this directory', str(current_path)))
                
                if not choices:
                    print(f"{Fore.RED}No accessible directories found!{Style.RESET_ALL}")
                    return None

                questions = [
                    inquirer.List('path',
                                message="Select directory or navigate",
                                choices=choices)
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

            except PermissionError:
                print(f"{Fore.RED}Permission denied accessing some directories{Style.RESET_ALL}")
                continue
            except Exception as e:
                logging.error(f"Error in directory navigation: {e}")
                print(f"{Fore.RED}Error accessing directory: {e}{Style.RESET_ALL}")
                return None

    def list_wav_files(self, directory):
        """List all WAV files recursively with size and date info"""
        wav_files = []
        try:
            for wav_path in directory.rglob("*.wav"):
                try:
                    size = wav_path.stat().st_size
                    modified = datetime.fromtimestamp(wav_path.stat().st_mtime)
                    wav_files.append({
                        'path': wav_path,
                        'size': size,
                        'modified': modified
                    })
                except Exception as e:
                    logging.warning(f"Could not get details for {wav_path}: {e}")
                    continue
        except Exception as e:
            logging.error(f"Error scanning directory {directory}: {e}")
            
        return wav_files

    def select_wav_file(self, wav_files):
        """Prompt user to select a WAV file with detailed information"""
        if not wav_files:
            print(f"{Fore.RED}No WAV files found in selected directory!{Style.RESET_ALL}")
            return None

        # Format file choices with size and date
        choices = []
        for wav in wav_files:
            size_mb = wav['size'] / (1024 * 1024)
            modified = wav['modified'].strftime("%Y-%m-%d %H:%M")
            display = f"{wav['path'].name} ({size_mb:.1f}MB, {modified})"
            choices.append((display, str(wav['path'])))

        questions = [
            inquirer.List('wav_file',
                         message="Select WAV file to add metadata",
                         choices=choices)
        ]
        
        answers = inquirer.prompt(questions)
        return answers['wav_file'] if answers else None

    def get_speaker_info(self):
        """Get detailed speaker information"""
        print(f"\n{Fore.CYAN}=== Speaker Information ==={Style.RESET_ALL}")
        
        # Get speaker count
        while True:
            try:
                count = input(f"{Fore.GREEN}Enter number of speakers (1-10): {Style.RESET_ALL}")
                count = int(count)
                if 1 <= count <= 10:
                    break
                print(f"{Fore.RED}Please enter a number between 1 and 10{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")

        # Get speaker names (optional)
        speakers = []
        print(f"\n{Fore.CYAN}Enter speaker names (optional, press Enter to use default names):{Style.RESET_ALL}")
        
        for i in range(count):
            default_name = f"Speaker_{i+1}"
            name = input(f"Speaker {i+1} name [{default_name}]: ").strip()
            speakers.append({
                'name': name or default_name,
                'id': i+1
            })

        return count, speakers

    def save_metadata(self, directory, wav_filename, speaker_count, speakers):
        """Save metadata in WhisperX compatible format with robust error handling"""
        wav_file = Path(wav_filename)  # wav_filename is now a full path
        logging.info(f"Processing metadata for {wav_file}")

        try:
            # Extract event details from filename
            filename_parts = wav_file.stem.split('_')
            if len(filename_parts) >= 6:  # audio_only_Event_Name_YYYYMMDD_HHMMSS
                date_str = filename_parts[-2]  # YYYYMMDD format
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                event_name = '_'.join(filename_parts[2:-2])
            else:
                event_name = wav_file.stem
                formatted_date = datetime.now().strftime("%Y-%m-%d")

            # Create comprehensive metadata
            metadata = {
                'title': event_name,
                'speaker_count': speaker_count,
                'date': formatted_date,
                'attendees': speakers,
                'file_name': wav_file.name,
                'date_modified': datetime.now().isoformat(),
                'metadata_version': '1.1',
                'file_info': {
                    'size_bytes': wav_file.stat().st_size,
                    'last_modified': datetime.fromtimestamp(wav_file.stat().st_mtime).isoformat()
                }
            }

            # Save metadata in all relevant locations
            metadata_locations = [
                wav_file.parent / f"{wav_file.stem}_metadata.json",  # Original location
                self.whisperx_workspace / f"{wav_file.stem}_metadata.json"  # WhisperX workspace
            ]

            success = True
            saved_locations = []
            
            for metadata_file in metadata_locations:
                try:
                    metadata_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    logging.info(f"Metadata saved to {metadata_file}")
                    saved_locations.append(metadata_file)
                except Exception as e:
                    logging.error(f"Error saving metadata to {metadata_file}: {e}")
                    print(f"{Fore.RED}Error saving to {metadata_file}: {e}{Style.RESET_ALL}")
                    success = False

            if saved_locations:
                print(f"\n{Fore.GREEN}✓ Metadata saved successfully to:{Style.RESET_ALL}")
                for location in saved_locations:
                    print(f"  • {location}")
            
            return success, metadata

        except Exception as e:
            logging.error(f"Error in save_metadata: {e}")
            print(f"\n{Fore.RED}Error processing metadata: {e}{Style.RESET_ALL}")
            return False, None

    def verify_metadata(self, wav_path):
        """Verify metadata file existence and contents"""
        print(f"\n{Fore.CYAN}=== Verifying Metadata ==={Style.RESET_ALL}")
        print(f"File: {wav_path.name}")

        metadata_locations = [
            wav_path.parent / f"{wav_path.stem}_metadata.json",
            self.whisperx_workspace / f"{wav_path.stem}_metadata.json"
        ]

        found_valid_metadata = False
        for metadata_file in metadata_locations:
            print(f"\nChecking {metadata_file}...")
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"{Fore.GREEN}✓ Found metadata file{Style.RESET_ALL}")
                    
                    # Verify required fields
                    required_fields = ['speaker_count', 'title', 'date', 'attendees']
                    missing_fields = [field for field in required_fields if field not in metadata]

                    if missing_fields:
                        print(f"{Fore.YELLOW}⚠ Missing fields: {', '.join(missing_fields)}{Style.RESET_ALL}")
                    else:
                        found_valid_metadata = True
                        print("\nMetadata contents:")
                        print(f"  • Title: {metadata['title']}")
                        print(f"  • Date: {metadata['date']}")
                        print(f"  • Speaker count: {metadata['speaker_count']}")
                        print("\nAttendees:")
                        for attendee in metadata['attendees']:
                            print(f"  • {attendee['name']}")

                except Exception as e:
                    print(f"{Fore.RED}Error reading metadata: {e}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}✗ No metadata file found{Style.RESET_ALL}")

        return found_valid_metadata

    def run(self):
        """Main program loop with error handling"""
        print(f"\n{Fore.CYAN}=== WAV File Metadata Manager for WhisperX ==={Style.RESET_ALL}")
        print(f"WhisperX Workspace: {self.whisperx_workspace}")

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
            wav_path = self.select_wav_file(wav_files)
            if not wav_path:
                return

            # Menu for actions
            questions = [
                inquirer.List('action',
                             message="Select action",
                             choices=[
                                 'Add/Update Metadata',
                                 'Verify Metadata',
                                 'Both',
                                 'Exit'
                             ])
            ]

            answers = inquirer.prompt(questions)
            if not answers:
                return
                
            action = answers['action']
            
            if action in ['Add/Update Metadata', 'Both']:
                speaker_count, speakers = self.get_speaker_info()
                success, metadata = self.save_metadata(selected_dir, wav_path, speaker_count, speakers)
                
                if success:
                    print(f"\n{Fore.GREEN}✓ Metadata successfully created/updated{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.RED}⚠ Some metadata operations failed{Style.RESET_ALL}")

            if action in ['Verify Metadata', 'Both']:
                if not self.verify_metadata(Path(wav_path)):
                    print(f"\n{Fore.RED}No valid metadata found in any location{Style.RESET_ALL}")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"\n{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")
            print("Check the logs for more details")

def main():
    manager = MetadataManager()
    manager.run()

if __name__ == "__main__":
    main()
