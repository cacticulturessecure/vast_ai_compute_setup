import os
import time
import inquirer
from pathlib import Path
from datetime import datetime
from colorama import init, Fore, Style
import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
from typing import Optional, List, Dict

# Initialize colorama
init()

class SpeakerVerificationCLI:
    def __init__(self):
        self.base_dir = Path("speaker_samples")
        self.sample_rate = 16000
        self.duration = 10  # seconds
        self.setup_logging()
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'speaker_verification.log'),
                logging.StreamHandler()
            ]
        )

    def record_audio(self, filename: Path) -> bool:
        """Record audio with countdown and visual feedback"""
        try:
            print(f"\n{Fore.CYAN}=== Recording Session ==={Style.RESET_ALL}")
            print("Get ready to speak...")
            
            # Countdown
            for i in range(3, 0, -1):
                print(f"{Fore.YELLOW}{i}...{Style.RESET_ALL}")
                time.sleep(1)
            
            print(f"{Fore.GREEN}🎤 Recording...{Style.RESET_ALL}")
            
            # Record audio
            audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            # Show progress bar
            for i in range(self.duration):
                progress = "=" * i + ">" + "." * (self.duration - i - 1)
                print(f"\r[{progress}] {i+1}/{self.duration}s", end="")
                time.sleep(1)
            print("\n")
            
            sd.wait()
            
            # Save the recording
            sf.write(filename, audio_data, self.sample_rate)
            print(f"{Fore.GREEN}✓ Recording saved to: {filename}{Style.RESET_ALL}")
            return True

        except Exception as e:
            logging.error(f"Error recording audio: {e}")
            print(f"\n{Fore.RED}❌ Error recording audio: {str(e)}{Style.RESET_ALL}")
            return False

    def enroll_speaker(self) -> Optional[str]:
        """Enroll a new speaker with voice sample"""
        print(f"\n{Fore.CYAN}=== Speaker Enrollment ==={Style.RESET_ALL}")
        
        # Get speaker name
        questions = [
            inquirer.Text('name',
                         message="Enter speaker name",
                         validate=lambda _, x: len(x.strip()) > 0)
        ]
        
        answers = inquirer.prompt(questions)
        if not answers:
            return None
            
        speaker_name = answers['name'].strip()
        
        # Create speaker directory
        speaker_dir = self.base_dir / speaker_name
        speaker_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = speaker_dir / f"enroll_{timestamp}.wav"
        
        print(f"\n{Fore.CYAN}Instructions:{Style.RESET_ALL}")
        print("1. Speak naturally for 10 seconds")
        print("2. Maintain consistent distance from microphone")
        print("3. Avoid background noise")
        
        input(f"\n{Fore.GREEN}Press Enter when ready to record...{Style.RESET_ALL}")
        
        if self.record_audio(filename):
            print(f"\n{Fore.GREEN}✓ Speaker {speaker_name} enrolled successfully!{Style.RESET_ALL}")
            return speaker_name
        return None

    def verify_speaker(self) -> bool:
        """Verify a speaker against enrolled samples"""
        print(f"\n{Fore.CYAN}=== Speaker Verification ==={Style.RESET_ALL}")
        
        # Get list of enrolled speakers
        enrolled_speakers = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
        if not enrolled_speakers:
            print(f"{Fore.YELLOW}No enrolled speakers found. Please enroll first.{Style.RESET_ALL}")
            return False
        
        # Select speaker to verify against
        questions = [
            inquirer.List('speaker',
                         message="Select speaker to verify against",
                         choices=enrolled_speakers)
        ]
        
        answers = inquirer.prompt(questions)
        if not answers:
            return False
            
        selected_speaker = answers['speaker']
        
        # Record verification sample
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.base_dir / selected_speaker / f"verify_{timestamp}.wav"
        
        input(f"\n{Fore.GREEN}Press Enter when ready to record verification sample...{Style.RESET_ALL}")
        
        if not self.record_audio(filename):
            return False
            
        # Simulate verification (placeholder)
        print(f"\n{Fore.YELLOW}Processing...{Style.RESET_ALL}")
        time.sleep(2)  # Simulate processing time
        
        # Placeholder result (to be implemented with actual verification logic)
        match_percentage = 85.5  # Example value
        threshold = 80.0
        is_match = match_percentage >= threshold
        
        # Display results
        print(f"\n{Fore.CYAN}=== Verification Results ==={Style.RESET_ALL}")
        print(f"Speaker: {selected_speaker}")
        print(f"Match Percentage: {match_percentage:.1f}%")
        
        if is_match:
            print(f"{Fore.GREEN}✓ PASS - Speaker verified{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ FAIL - Speaker not verified{Style.RESET_ALL}")
            
        return is_match

    def run(self):
        """Main program loop"""
        while True:
            print(f"\n{Fore.CYAN}=== Speaker Verification System ==={Style.RESET_ALL}")
            print("1. Enroll New Speaker")
            print("2. Verify Speaker")
            print("3. Exit")
            
            questions = [
                inquirer.List('action',
                             message="Select action",
                             choices=['Enroll New Speaker', 'Verify Speaker', 'Exit'])
            ]
            
            answers = inquirer.prompt(questions)
            if not answers:
                break
                
            action = answers['action']
            
            if action == 'Enroll New Speaker':
                self.enroll_speaker()
            elif action == 'Verify Speaker':
                self.verify_speaker()
            else:
                print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                break

def main():
    try:
        verifier = SpeakerVerificationCLI()
        verifier.run()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Program interrupted by user.{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"\n{Fore.RED}An unexpected error occurred: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
