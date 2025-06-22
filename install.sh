#!/bin/bash
# OmniParser Installation Script
# Auto-detects platform and installs appropriate dependencies

set -e  # Exit on any error

echo "🚀 OmniParser Installation Script"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Run the auto-detection installer
echo "🔍 Running platform-specific installer..."
python install_dependencies.py

echo ""
echo "✅ Installation completed!"
echo ""
echo "To activate the environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To run OmniParser:"
echo "  python batch_run_omniparser.py" 