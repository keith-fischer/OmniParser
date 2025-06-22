# OmniParser Installation Guide

This guide covers installation for different platforms with automatic platform detection and optimized dependencies.

## 🚀 Quick Start

### One-Command Installation

**Mac/Linux:**
```bash
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
install.bat
```

## 🔧 Platform-Specific Details

### Mac Installation
- **Auto-detection**: Detects Apple Silicon (M1/M2) vs Intel
- **Optimizations**: Uses native PyTorch builds for optimal performance
- **Dependencies**: `requirements_mac.txt` contains Mac-optimized versions

### Windows Installation
- **CUDA Support**: Automatically installs PyTorch with CUDA 11.8
- **GPU Acceleration**: Enables NVIDIA GPU acceleration when available
- **Dependencies**: `requirements_windows.txt` includes Windows-specific packages
- **Prerequisites**: 
  - NVIDIA GPU drivers
  - CUDA Toolkit (optional, for GPU acceleration)

### Linux Installation
- **Generic Setup**: Uses standard `requirements.txt`
- **Flexibility**: Works with various Linux distributions
- **GPU Support**: CUDA support available with proper drivers

## 📦 Manual Installation Options

### Option 1: Auto-Detection Script
```bash
python install_dependencies.py
```
**Features:**
- Automatically detects your platform
- Installs appropriate dependencies
- Handles CUDA setup for Windows
- Verifies installation success

**Options:**
```bash
# Force specific platform
python install_dependencies.py --force-platform windows

# Skip PyTorch installation (if already installed)
python install_dependencies.py --skip-pytorch

# Verbose output
python install_dependencies.py --verbose
```

### Option 2: Platform-Specific Requirements

**Mac:**
```bash
pip install -r requirements_mac.txt
```

**Windows:**
```bash
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Then install other dependencies
pip install -r requirements_windows.txt
```

**Linux:**
```bash
pip install -r requirements.txt
```

### Option 3: Virtual Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate.bat

# Install dependencies
python install_dependencies.py
```

## 🔍 Verification

After installation, verify everything works:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import cv2
print('OpenCV: OK')

import transformers
print('Transformers: OK')

print('✅ All dependencies installed successfully!')
"
```

## 🎯 Running OmniParser

After installation:

```bash
# Activate environment (if using virtual environment)
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate.bat  # Windows

# Run the batch processor
python batch_run_omniparser.py

# Run the Gradio demo
python gradio_demo.py
```

## 🛠️ Troubleshooting

### Common Issues

**CUDA Installation Fails (Windows):**
```bash
# Try CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Permission Errors (Mac/Linux):**
```bash
# Use sudo if needed
sudo python install_dependencies.py
```

**Path Issues (Windows):**
- Ensure Python is in your PATH
- Use full paths if needed: `C:\Python39\python.exe install_dependencies.py`

**Virtual Environment Issues:**
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python install_dependencies.py
```

### Platform-Specific Notes

**Mac:**
- Works with both Intel and Apple Silicon
- No additional drivers needed
- Uses native PyTorch builds

**Windows:**
- Requires NVIDIA drivers for GPU acceleration
- CUDA Toolkit recommended for best performance
- May need Visual Studio Build Tools for some packages

**Linux:**
- Works with most distributions
- May need additional system packages (gcc, g++, etc.)
- CUDA support varies by distribution

## 📋 Requirements Files Overview

- `requirements_mac.txt`: Mac-optimized dependencies
- `requirements_windows.txt`: Windows-specific with CUDA support
- `requirements.txt`: Generic requirements (used for Linux)
- `install_dependencies.py`: Auto-detection and installation script
- `install.sh`: Mac/Linux installation wrapper
- `install.bat`: Windows installation wrapper

## 🎉 Next Steps

After successful installation:

1. **Download Model Weights:**
   ```bash
   # Download V2 weights
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do 
     huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
   done
   mv weights/icon_caption weights/icon_caption_florence
   ```

2. **Test Installation:**
   ```bash
   python gradio_demo.py
   ```

3. **Run Batch Processing:**
   ```bash
   python batch_run_omniparser.py
   ```

For more information, see the main [README.md](README.md) file. 