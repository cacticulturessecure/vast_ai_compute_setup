import os
import subprocess
from pathlib import Path
import inquirer
import time

class VastAIFileDownloader:
    def __init__(self):
        self.host = None
        self.port = None
        self.remote_base_path = None
        self.local_download_path = None

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

    def get_remote_directories(self):
        """Get list of directories from vast.ai instance"""
        try:
            # Use ls -l to get detailed listing and grep to filter only directories
            cmd = f"ssh -p {self.port} {self.host} 'ls -l {self.remote_base_path} | grep ^d'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse the output to get directory names
            directories = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    # Split by spaces and get the last part (directory name)
                    dir_name = line.split()[-1]
                    directories.append(dir_name)
            
            return directories
        except subprocess.CalledProcessError as e:
            print(f"Error getting remote directories: {str(e)}")
            return []

    def select_download_location(self):
        """Let user select or create download location"""
        questions = [
            inquirer.Text('download_path', 
                         message="Enter local download path (press Enter for current directory)",
                         default=str(Path.cwd()))
        ]
        
        answers = inquirer.prompt(questions)
        download_path = Path(answers['download_path'])
        
        # Create directory if it doesn't exist
        download_path.mkdir(parents=True, exist_ok=True)
        self.local_download_path = download_path
        return download_path

    def select_remote_items(self, directories):
        """Let user select which remote directories to download"""
        if not directories:
            print("No directories found in remote location!")
            return []

        questions = [
            inquirer.Checkbox('directories',
                            message="Select directories to download (use spacebar to select, enter to confirm)",
                            choices=directories)
        ]
        
        answers = inquirer.prompt(questions)
        return answers['directories']

    def download_files(self, selected_directories):
        """Download selected directories from vast.ai"""
        if not selected_directories:
            print("No directories selected for download.")
            return

        for directory in selected_directories:
            print(f"\nDownloading {directory}...")
            
            # Create local directory
            local_dir = self.local_download_path / directory
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct the scp command
            remote_path = f"{self.remote_base_path}/{directory}"
            scp_cmd = f"scp -r -P {self.port} {self.host}:{remote_path}/* {local_dir}/"
            
            try:
                result = subprocess.run(scp_cmd, shell=True, check=True)
                if result.returncode == 0:
                    print(f"✅ Successfully downloaded {directory}")
                else:
                    print(f"❌ Failed to download {directory}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error downloading {directory}: {str(e)}")

    def run(self):
        """Main execution method"""
        print("🚀 VAST.AI File Download Utility")
        print("===============================")
        
        # Get vast.ai connection details
        self.get_vast_ai_credentials()
        
        # Get remote directories
        print("\nFetching remote directories...")
        remote_directories = self.get_remote_directories()
        
        if remote_directories:
            print("\nFound the following remote directories:")
            for directory in remote_directories:
                print(f"  📁 {directory}")
            
            # Select download location
            print("\nSelect download location:")
            download_path = self.select_download_location()
            print(f"Files will be downloaded to: {download_path}")
            
            # Select directories to download
            print("\nSelect directories to download:")
            selected_directories = self.select_remote_items(remote_directories)
            
            if selected_directories:
                print("\nSelected directories for download:")
                for directory in selected_directories:
                    print(f"  📁 {directory}")
                
                # Confirm download
                questions = [
                    inquirer.Confirm('confirm',
                                   message='Do you want to proceed with the download?',
                                   default=True)
                ]
                
                if inquirer.prompt(questions)['confirm']:
                    self.download_files(selected_directories)
                    print("\n✨ Download process completed!")
                else:
                    print("\n❌ Download cancelled by user.")
            else:
                print("\n❌ No directories selected for download.")
        else:
            print("\n❌ No remote directories found.")

def main():
    try:
        downloader = VastAIFileDownloader()
        downloader.run()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
