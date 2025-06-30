# Omniparser Modularization

This document describes the modularized structure of the Omniparser project, where the single image processing logic has been encapsulated into `omniparser.py` and the batch processing functionality is separated into `batch_run_omniparser.py`.

## Project Structure

### Core Modules

#### `util/omniparser.py`
The main Omniparser class that encapsulates all single image processing functionality.

**Key Features:**
- **Initialization**: Can be initialized with default or custom configuration
- **Multiple Input Formats**: Supports file paths, PIL Images, and base64 strings
- **Flexible Output**: Can save files or return results only
- **Comprehensive Results**: Returns detailed processing information

**Main Methods:**
- `__init__(config=None)`: Initialize with optional custom configuration
- `parse_base64(image_base64)`: Process base64 encoded image
- `parse_image(image)`: Process PIL Image object
- `process_single_image(image_path, output_dir=None, datetime_prefix=None, save_outputs=True)`: Process image file with optional file saving

#### `batch_run_omniparser.py`
Focused solely on batch processing functionality.

**Key Features:**
- **File Discovery**: Finds and processes multiple images in a directory
- **Batch Configuration**: Creates Omniparser instances with specific parameters
- **Progress Tracking**: Shows progress and handles errors gracefully
- **Summary Reports**: Generates comprehensive batch processing reports

**Main Functions:**
- `process_batch()`: Process all images in a directory
- `save_batch_summary()`: Save batch processing summary
- `create_ascii_table_report()`: Generate formatted reports
- `run_omniparse()`: Main batch processing function

### Example Usage

#### Single Image Processing

```python
from util.omniparser import Omniparser
from pathlib import Path

# Basic usage with default configuration
omniparser = Omniparser()

# Process single image and save outputs
result = omniparser.process_single_image(
    image_path=Path("./imgs/example.png"),
    output_dir=Path("./output"),
    save_outputs=True
)

print(f"Detected {result['num_elements']} elements")
print(f"Found {result['num_ocr_text']} OCR text items")
```

#### Custom Configuration

```python
# Custom configuration
config = {
    'BOX_TRESHOLD': 0.25,  # Higher threshold for fewer detections
    'iou_threshold': 0.3,  # Higher IoU threshold
    'use_paddleocr': False,  # Use EasyOCR instead
    'imgsz': 800,  # Different image size
}

omniparser = Omniparser(config)
result = omniparser.process_single_image(image_path, save_outputs=False)
```

#### Base64 Processing

```python
# Process base64 encoded image
annotated_image_base64, parsed_content = omniparser.parse_base64(base64_string)
```

#### PIL Image Processing

```python
from PIL import Image

# Process PIL Image directly
pil_image = Image.open("example.png")
annotated_image_base64, parsed_content = omniparser.parse_image(pil_image)
```

#### Batch Processing

```python
# Command line usage
python batch_run_omniparser.py --input_dir ./images --output_dir ./output --box 0.1 --iou 0.1

# Programmatic usage
from batch_run_omniparser import run_omniparse

results = run_omniparse(
    img_path="./imgs",
    img_filter="*.png",
    out_dir="./output",
    box=0.1,
    iou=0.1,
    img_size=3000
)
```

## Configuration Options

### Omniparser Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `som_model_path` | `'weights/icon_detect/model.pt'` | Path to YOLO model |
| `caption_model_name` | `'florence2'` | Caption model name |
| `caption_model_path` | `'weights/icon_caption_florence'` | Path to caption model |
| `BOX_TRESHOLD` | `0.15` | Confidence threshold for bounding box detection |
| `iou_threshold` | `0.15` | IoU threshold for non-maximum suppression |
| `use_paddleocr` | `True` | Whether to use PaddleOCR instead of EasyOCR |
| `imgsz` | `640` | Input image size for YOLO model |
| `use_local_semantics` | `True` | Use local semantics processing |
| `scale_img` | `False` | Scale image during processing |
| `batch_size` | `128` | Batch size for processing |

### Batch Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | `./imgs` | Input directory containing images |
| `--output_dir` | `./output` | Output directory for results |
| `--img_filter` | `*.png` | File filter pattern |
| `--box` | `0.05` | Box threshold for detection |
| `--iou` | `0.05` | IoU threshold for detection |
| `--img_size` | `3000` | Input image size for YOLO model |

## Output Structure

### Single Image Processing Output

```python
{
    "image_name": "example",
    "image_size": (1920, 1080),
    "num_elements": 15,
    "num_ocr_text": 8,
    "detected_elements": [...],  # List of detected UI elements
    "label_coordinates": {...},  # Bounding box coordinates
    "ocr_text": [...],  # List of detected text
    "ocr_bbox": [...],  # OCR bounding boxes
    "annotated_image_base64": "...",  # Base64 encoded annotated image
    "processing_parameters": {...},  # Processing parameters used
    # File paths (if save_outputs=True):
    "original_copy_path": "path/to/original_copy.png",
    "marked_up_path": "path/to/marked_up.png",
    "json_path": "path/to/elements.json"
}
```

### Batch Processing Output

The batch processor creates:
1. **Individual Image Directories**: Each image gets its own subdirectory
2. **Three Files per Image**:
   - Original image copy
   - Annotated image with detected regions
   - JSON file with all detected elements
3. **Batch Summary**: Overall processing summary in JSON format
4. **Console Reports**: Formatted ASCII table reports

## Benefits of Modularization

1. **Separation of Concerns**: Single image processing is separate from batch processing
2. **Reusability**: Omniparser class can be used in other contexts (web APIs, GUI apps, etc.)
3. **Testability**: Each module can be tested independently
4. **Maintainability**: Easier to modify and extend individual components
5. **Flexibility**: Multiple input formats and output options
6. **Configuration**: Easy to customize processing parameters

## Migration Guide

### From Old Structure

If you were using the old `batch_run_omniparser.py` for single image processing:

**Old way:**
```python
# Had to use batch processing for single images
from batch_run_omniparser import process_single_image
result = process_single_image(image_path, output_dir, datetime_prefix, ...)
```

**New way:**
```python
# Direct single image processing
from util.omniparser import Omniparser
omniparser = Omniparser()
result = omniparser.process_single_image(image_path, output_dir)
```

### Examples

See `example_single_image.py` for comprehensive usage examples.

## Error Handling

Both modules include comprehensive error handling:

- **Single Image**: Errors are caught and returned in the result dictionary
- **Batch Processing**: Errors are logged and processing continues with other images
- **Model Loading**: Graceful fallback to CPU if CUDA is not available
- **File Operations**: Safe file handling with proper error messages

## Performance Considerations

- **Model Loading**: Models are loaded once during initialization
- **Memory Management**: Images are processed one at a time to manage memory
- **Batch Size**: Configurable batch size for caption model processing
- **Image Size**: Configurable input size for YOLO model 