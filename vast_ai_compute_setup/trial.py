import argparse
import sys

def display_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            # Print with clear delimitation
            print("\n" + "="*80)
            print(content)
            print("="*80 + "\n")
            print(f"Above are the contents of '{filename}'")
            print("You can select and copy the content between the line markers")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Display file contents for easy copying')
    parser.add_argument('--filename', required=True, help='Path to the file to be displayed')
    
    args = parser.parse_args()
    display_file(args.filename)

if __name__ == "__main__":
    main()
