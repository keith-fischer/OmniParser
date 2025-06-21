#!/bin/bash

echo "Installing OmniParser dependencies..."

# Upgrade pip first
pip install --upgrade pip

# Install numpy first (required for pandas)
echo "Installing numpy..."
pip install numpy==1.26.4

# Install pandas using pre-compiled wheel
echo "Installing pandas..."
pip install pandas>=2.0.0

# Install tokenizers using pre-compiled wheel
echo "Installing tokenizers..."
pip install tokenizers>=0.15.0

# Install PyTorch and related packages
echo "Installing PyTorch and related packages..."
pip install torch torchvision

# Install the rest of the requirements
echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo "Installation complete!" 