import argparse
import pyperclip
import sys

def copy_file_to_clipboard(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            pyperclip.copy(content)
            print(f"Contents of '{filename}' copied to clipboard successfully!")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Copy file contents to clipboard')
    parser.add_argument('--filename', required=True, help='Path to the file to be copied')
    
    args = parser.parse_args()
    copy_file_to_clipboard(args.filename)

if __name__ == "__main__":
    main()
