import os
import subprocess
from pathlib import Path
import inquirer
import time

class VastAIFileTransfer:
    def __init__(self):
        self.host = None
        self.port = None
        self.remote_base_path = None

    def get_vast_ai_credentials(self):
        """Get vast.ai connection details from user"""
        questions = [
            inquirer.Text('host', message="Enter vast.ai host (e.g., root@24.52.17.82)"),
            inquirer.Text('port', message="Enter vast.ai port (e.g., 48834)"),
            inquirer.Text('remote_base_path', message="Enter remote base path (e.g., /workspace/audio)")
        ]
        answers = inquirer.prompt(questions)
        
        self.host = answers['host']
        self.port = answers['port']
        self.remote_base_path = answers['remote_base_path']

    def select_local_folders(self):
        """Let user select local folders to transfer"""
        # Get current directory contents
        current_dir = Path.cwd()
        items = [item for item in current_dir.iterdir() if item.is_dir()]
        
        if not items:
            print("No directories found in current location!")
            return []

        questions = [
            inquirer.Checkbox('folders',
                            message="Select folders to transfer (use spacebar to select, enter to confirm)",
                            choices=[str(item.name) for item in items])
        ]
        
        answers = inquirer.prompt(questions)
        return answers['folders']

    def transfer_files(self, selected_folders):
        """Transfer selected folders to vast.ai"""
        if not selected_folders:
            print("No folders selected for transfer.")
            return

        # Create remote directory if it doesn't exist
        mkdir_cmd = f"ssh -p {self.port} {self.host} 'mkdir -p {self.remote_base_path}'"
        subprocess.run(mkdir_cmd, shell=True)

        for folder in selected_folders:
            print(f"\nTransferring {folder}...")
            
            # Create the full remote path
            remote_path = f"{self.remote_base_path}/{folder}"
            
            # Create the specific folder in remote
            mkdir_cmd = f"ssh -p {self.port} {self.host} 'mkdir -p {remote_path}'"
            subprocess.run(mkdir_cmd, shell=True)

            # Transfer the files
            scp_cmd = f"scp -r -P {self.port} ./{folder}/* {self.host}:{remote_path}/"
            
            try:
                result = subprocess.run(scp_cmd, shell=True, check=True)
                if result.returncode == 0:
                    print(f"✅ Successfully transferred {folder}")
                else:
                    print(f"❌ Failed to transfer {folder}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error transferring {folder}: {str(e)}")

    def run(self):
        """Main execution method"""
        print("🚀 VAST.AI File Transfer Utility")
        print("===============================")
        
        # Get vast.ai connection details
        self.get_vast_ai_credentials()
        
        # Select folders to transfer
        print("\nSelect folders to transfer:")
        selected_folders = self.select_local_folders()
        
        if selected_folders:
            print("\nSelected folders for transfer:")
            for folder in selected_folders:
                print(f"  📁 {folder}")
            
            # Confirm transfer
            questions = [
                inquirer.Confirm('confirm',
                               message='Do you want to proceed with the transfer?',
                               default=True)
            ]
            
            if inquirer.prompt(questions)['confirm']:
                self.transfer_files(selected_folders)
                print("\n✨ Transfer process completed!")
            else:
                print("\n❌ Transfer cancelled by user.")
        else:
            print("\n❌ No folders selected for transfer.")

def main():
    try:
        transfer = VastAIFileTransfer()
        transfer.run()
    except KeyboardInterrupt:
        print("\n\n❌ Transfer cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
