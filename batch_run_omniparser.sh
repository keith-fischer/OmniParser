#!/bin/bash

# =============================================================================
# OmniParser Batch Processing Script
# =============================================================================
# 
# This script runs the OmniParser batch processing with configurable parameters.
# Edit the variables below to customize the processing behavior.
#
# Usage:
#   ./batch_run_omniparser.sh
#   chmod +x batch_run_omniparser.sh  # Make executable first time
# =============================================================================

# =============================================================================
# CONFIGURATION VARIABLES - EDIT THESE AS NEEDED
# =============================================================================

# Input and Output Directories
INPUT_DIR="./imgs/tsserrors"           # Directory containing PNG images to process
OUTPUT_DIR="./output/tsserrors"        # Directory to save processing results

# File Filter
IMG_FILTER="*.png"                     # File pattern to process (e.g., "*.png", "*.jpg", "frame_*.png")

# Detection Parameters
BOX_THRESHOLD="0.05"                   # Box threshold for detection (0.01-0.5, lower = more sensitive)
IOU_THRESHOLD="0.05"                   # IoU threshold for detection (0.01-0.5, lower = more sensitive)
IMG_SIZE="3000"                        # Input image size for YOLO model (640, 1000, 2000, 3000, etc.)

# OCR Engine Selection
USE_PADDLEOCR="false"                  # Use PaddleOCR instead of EasyOCR (true/false)

# Processing Options
SAVE_OUTPUTS="true"                    # Save output files (true/false)
VERBOSE="true"                         # Verbose output (true/false)

# =============================================================================
# ADVANCED CONFIGURATION (usually don't need to change)
# =============================================================================

# Python Environment
PYTHON_CMD="python"                    # Python command (python, python3, etc.)
SCRIPT_PATH="batch_run_omniparser.py"  # Path to the Python script

# Logging
LOG_FILE=""                            # Log file path (empty = no logging)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")     # Timestamp for this run

# =============================================================================
# SCRIPT EXECUTION - DON'T EDIT BELOW THIS LINE
# =============================================================================

echo "============================================================================="
echo "OmniParser Batch Processing"
echo "============================================================================="
echo "Timestamp: $TIMESTAMP"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "File Filter: $IMG_FILTER"
echo "Box Threshold: $BOX_THRESHOLD"
echo "IoU Threshold: $IOU_THRESHOLD"
echo "Image Size: $IMG_SIZE"
echo "Use PaddleOCR: $USE_PADDLEOCR"
echo "============================================================================="

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory '$INPUT_DIR' does not exist!"
    echo "Please check the INPUT_DIR variable at the top of this script."
    exit 1
fi

# Check if Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: Python script '$SCRIPT_PATH' not found!"
    echo "Please check the SCRIPT_PATH variable at the top of this script."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create logs directory if logging is enabled
if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
fi

# Build command arguments
CMD_ARGS=(
    "$PYTHON_CMD"
    "$SCRIPT_PATH"
    "--input_dir" "$INPUT_DIR"
    "--output_dir" "$OUTPUT_DIR"
    "--img_filter" "$IMG_FILTER"
    "--box" "$BOX_THRESHOLD"
    "--iou" "$IOU_THRESHOLD"
    "--img_size" "$IMG_SIZE"
)

# Execute the command
echo "🚀 Starting batch processing..."
echo "Command: ${CMD_ARGS[*]}"
echo ""

# Run with or without logging
if [ -n "$LOG_FILE" ]; then
    echo "📝 Logging output to: $LOG_FILE"
    "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
else
    "${CMD_ARGS[@]}"
    EXIT_CODE=$?
fi

echo ""
echo "============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Batch processing completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
else
    echo "❌ Batch processing failed with exit code: $EXIT_CODE"
fi
echo "============================================================================="

exit $EXIT_CODE 