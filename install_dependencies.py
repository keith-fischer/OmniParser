#!/usr/bin/env python3
"""
OmniParser Dependency Installer
Auto-detects platform and installs appropriate dependencies
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path


def detect_platform():
    """Detect the current platform and return platform info"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin":
        # Check if it's Apple Silicon
        if machine in ["arm64", "aarch64"]:
            return "mac_arm64"  # Apple Silicon (M1/M2)
        else:
            return "mac_intel"  # Intel Mac
    elif system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"


def check_cuda_availability():
    """Check if CUDA is available on the system"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def install_pytorch_with_cuda():
    """Install PyTorch with CUDA support for Windows"""
    print("🔧 Installing PyTorch with CUDA support...")
    try:
        # Install PyTorch with CUDA 11.8 (adjust version as needed)
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)
        print("✅ PyTorch with CUDA installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch with CUDA: {e}")
        return False


def install_requirements(requirements_file):
    """Install requirements from a specific file"""
    print(f"📦 Installing requirements from {requirements_file}...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], check=True)
        print(f"✅ Requirements installed successfully from {requirements_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def setup_environment():
    """Set up the Python environment"""
    print("🐍 Setting up Python environment...")
    
    # Upgrade pip
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        print("✅ Pip upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to upgrade pip: {e}")


def main():
    parser = argparse.ArgumentParser(description="OmniParser Dependency Installer")
    parser.add_argument("--force-platform", choices=["mac", "windows", "linux"], 
                       help="Force installation for specific platform")
    parser.add_argument("--skip-pytorch", action="store_true",
                       help="Skip PyTorch installation (useful if already installed)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🚀 OmniParser Dependency Installer")
    print("=" * 50)
    
    # Detect platform
    if args.force_platform:
        platform_type = args.force_platform
        print(f"🔧 Using forced platform: {platform_type}")
    else:
        platform_type = detect_platform()
        print(f"🔍 Detected platform: {platform_type}")
    
    # Setup environment
    setup_environment()
    
    # Install platform-specific requirements
    if platform_type.startswith("mac"):
        requirements_file = "requirements_mac.txt"
        print("🍎 Installing Mac-specific dependencies...")
    elif platform_type == "windows":
        requirements_file = "requirements_windows.txt"
        print("🪟 Installing Windows-specific dependencies...")
        
        # Install PyTorch with CUDA for Windows (unless skipped)
        if not args.skip_pytorch:
            if not install_pytorch_with_cuda():
                print("⚠️  CUDA installation failed, continuing with CPU-only PyTorch...")
    elif platform_type == "linux":
        requirements_file = "requirements.txt"  # Use generic requirements for Linux
        print("🐧 Installing Linux dependencies...")
    else:
        print("❌ Unsupported platform detected")
        sys.exit(1)
    
    # Check if requirements file exists
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file {requirements_file} not found")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements(requirements_file):
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (using CPU)")
        
        import cv2
        print("✅ OpenCV installed")
        
        import transformers
        print("✅ Transformers installed")
        
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Activate your virtual environment")
        print("2. Run: python batch_run_omniparser.py")
        
    except ImportError as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 