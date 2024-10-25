#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
  echo -e "${YELLOW}[STATUS]${NC} $1"
}

# Function to print success messages
print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to print error messages
print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command was successful
check_status() {
  if [ $? -eq 0 ]; then
    print_success "$1"
  else
    print_error "$2"
    exit 1
  fi
}

# Start setup
print_status "Starting package installation..."

# Create log file
LOG_FILE="setup_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE") 2>&1

print_status "Log file created at $LOG_FILE"

# Update package lists
print_status "Updating package lists..."
apt-get update
check_status "Package lists updated successfully" "Failed to update package lists"

# Install essential packages first
print_status "Installing essential packages..."
apt-get install -y \
  build-essential \
  software-properties-common \
  curl \
  wget \
  git \
  python3-pip \
  python3-dev
check_status "Essential packages installed successfully" "Failed to install essential packages"

# Install system packages from apt_packages.txt
print_status "Installing system packages from apt_packages.txt..."
while read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" =~ ^#.*$ ]] && continue

  # Extract package name (second column from dpkg -l output)
  package_name=$(echo "$line" | awk '{print $2}')

  if [[ ! -z "$package_name" ]]; then
    print_status "Installing $package_name..."
    apt-get install -y "$package_name" || print_error "Failed to install $package_name"
  fi
done <apt_packages.txt
check_status "System packages installed successfully" "Some system packages failed to install"

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip
check_status "Pip upgraded successfully" "Failed to upgrade pip"

# Install Python packages from requirements.txt
print_status "Installing Python packages from requirements.txt..."
pip install -r requirements.txt
check_status "Python packages installed successfully" "Failed to install Python packages"

# Final system update
print_status "Performing final system update..."
apt-get update && apt-get upgrade -y
check_status "Final system update completed successfully" "Failed to perform final system update"

# Create a validation file
print_status "Creating validation file..."
{
  echo "Setup completed on: $(date)"
  echo "Python version: $(python3 --version)"
  echo "Pip version: $(pip --version)"
  echo "CUDA version: $(nvidia-smi | grep "CUDA Version" || echo "CUDA not found")"
} >setup_validation.txt

print_success "Package installation completed successfully!"
print_status "Check setup_validation.txt for setup details"
print_status "Check $LOG_FILE for complete setup log"

print_success "Setup complete! You may need to log out and back in for all changes to take effect."
