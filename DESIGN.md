# OmniParser Project Design Document

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Entry Points](#entry-points)
4. [Core Modules](#core-modules)
5. [Dependencies and External Libraries](#dependencies-and-external-libraries)
6. [Data Flow](#data-flow)
7. [Configuration Management](#configuration-management)
8. [Error Handling](#error-handling)
9. [Performance Considerations](#performance-considerations)
10. [Future Enhancements](#future-enhancements)

## 🎯 Project Overview

OmniParser is a comprehensive screen parsing tool designed for GUI agent applications. It processes screenshots to detect and classify UI elements, extract text via OCR, and generate structured representations of user interfaces.

### **Key Capabilities:**
- **Icon Detection**: YOLO-based detection of UI elements and icons
- **Text Extraction**: Multi-engine OCR with automatic fallback
- **Element Classification**: AI-powered captioning of detected elements
- **Batch Processing**: Scalable processing of multiple images
- **Structured Output**: JSON metadata with annotated images

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                             │
├─────────────────────────────────────────────────────────────────┤
│  batch_run_omniparser.sh  │  batch_run_omniparser.py           │
│  (Shell Script)           │  (Python CLI)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Processing Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  util/omniparser.py  │  Main Omniparser Class                  │
│  - Single image processing                                      │
│  - Configuration management                                     │
│  - Output generation                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Utility Modules                              │
├─────────────────────────────────────────────────────────────────┤
│  util/utils.py      │  Core processing functions               │
│  util/ocr_engine.py │  Three-tier OCR system                   │
│  util/box_annotator.py │ Visualization utilities               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Dependencies                        │
├─────────────────────────────────────────────────────────────────┤
│  YOLO Models │  OCR Engines │  Caption Models │  Image Processing │
│  - icon_detect │  - EasyOCR │  - Florence2   │  - OpenCV        │
│  - model.pt   │  - PaddleOCR│  - BLIP2       │  - PIL/Pillow    │
│              │  - Tesseract │                │  - NumPy         │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Entry Points

### **1. batch_run_omniparser.sh**
**Purpose**: User-friendly shell script for batch processing
**Features**:
- Configurable variables at the top for easy customization
- Error checking and validation
- Progress reporting and logging
- Cross-platform compatibility

**Configuration Variables**:
```bash
INPUT_DIR="./imgs/tsserrors"           # Input directory
OUTPUT_DIR="./output/tsserrors"        # Output directory
IMG_FILTER="*.png"                     # File pattern
BOX_THRESHOLD="0.05"                   # Detection sensitivity
IOU_THRESHOLD="0.05"                   # IoU threshold
IMG_SIZE="3000"                        # YOLO input size
USE_PADDLEOCR="false"                  # OCR engine selection
```

**Flow**:
1. Validate input directory and script existence
2. Create output directory if needed
3. Build command arguments from variables
4. Execute `batch_run_omniparser.py` with parameters
5. Handle logging and error reporting

### **2. batch_run_omniparser.py**
**Purpose**: Python CLI for batch processing
**Features**:
- Command-line argument parsing
- Batch processing orchestration
- Result aggregation and reporting
- Error handling and recovery

**Key Functions**:
- `run_omniparse()`: Main processing function
- `process_batch()`: Batch processing logic
- `save_batch_summary()`: Result persistence
- `create_ascii_table_report()`: Human-readable output

## 🔧 Core Modules

### **1. util/omniparser.py**
**Purpose**: Main Omniparser class for single image processing
**Key Components**:

#### **Omniparser Class**
```python
class Omniparser:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def process_single_image(self, image_path, output_dir, datetime_prefix, save_outputs)
    def parse_base64(self, image_base64: str)
    def parse_image(self, image: Image.Image)
    def test_module(self, test_image_path, save_test_output)
```

**Responsibilities**:
- **Configuration Management**: Load and validate processing parameters
- **Model Initialization**: Load YOLO and caption models
- **Image Processing**: Orchestrate the complete processing pipeline
- **Output Generation**: Create annotated images and JSON metadata
- **Testing**: Comprehensive module testing functionality

**Dependencies**:
- `util.utils`: Core processing functions
- `util.ocr_engine`: OCR functionality
- PyTorch: Model inference
- PIL: Image handling

### **2. util/utils.py**
**Purpose**: Core processing functions and utilities
**Key Functions**:

#### **Model Management**
```python
def get_yolo_model(model_path)
def get_caption_model_processor(model_name, model_name_or_path, device)
```

#### **Image Processing**
```python
def get_som_labeled_img(image_source, model, BOX_TRESHOLD, ...)
def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold)
def annotate(image_source, boxes, logits, phrases, text_scale, ...)
```

#### **OCR Integration**
```python
def check_ocr_box(image_source, display_img, output_bb_format, ...)
```

**Dependencies**:
- `util.ocr_engine`: OCR functionality
- `util.box_annotator`: Visualization
- Ultralytics: YOLO model inference
- Transformers: Caption model processing
- OpenCV: Image processing
- NumPy: Array operations

### **3. util/ocr_engine.py**
**Purpose**: Three-tier OCR system with automatic fallback
**Key Components**:

#### **OCREngine Class**
```python
class OCREngine:
    def __init__(self, primary_engine='easyocr', enable_fallbacks=True, ...)
    def extract_text(self, image_source, output_format='xyxy', **kwargs)
    def test_engines(self, test_image_path)
```

**Engine Hierarchy**:
1. **EasyOCR** (Primary): Best balance of accuracy and ease of use
2. **PaddleOCR** (Secondary): High accuracy, good for complex layouts
3. **Tesseract** (Tertiary): Reliable fallback for clean documents

**Features**:
- Automatic fallback when primary engine fails
- Performance monitoring and timing
- Detailed metadata about which engine was used
- Backward compatibility with existing code

**Dependencies**:
- EasyOCR: Primary OCR engine
- PaddleOCR: Secondary OCR engine
- pytesseract: Tertiary OCR engine
- PIL: Image handling
- NumPy: Array operations

### **4. util/box_annotator.py**
**Purpose**: Visualization utilities for bounding box annotation
**Key Components**:

#### **BoxAnnotator Class**
```python
class BoxAnnotator:
    def __init__(self, color, thickness, text_color, text_scale, ...)
    def annotate(self, scene, detections, labels, skip_label, image_size)
```

**Features**:
- Bounding box drawing with customizable styles
- Text label placement with overlap avoidance
- Color palette support
- Optimal label positioning algorithms

**Dependencies**:
- OpenCV: Image drawing and text rendering
- NumPy: Array operations
- Supervision: Detection data structures

## 📦 Dependencies and External Libraries

### **Core Dependencies**
```python
# Deep Learning & Computer Vision
torch                    # PyTorch for model inference
torchvision             # Computer vision utilities
ultralytics             # YOLO model framework
transformers            # Hugging Face models (Florence2, BLIP2)
opencv-python           # OpenCV for image processing
supervision             # Detection visualization

# Image Processing
Pillow                  # PIL for image handling
numpy                   # Numerical computing

# OCR Engines
easyocr                 # Primary OCR engine
paddleocr               # Secondary OCR engine
pytesseract             # Tertiary OCR engine

# Utilities
matplotlib              # Plotting and visualization
requests                # HTTP requests
```

### **Model Dependencies**
```
weights/
├── icon_detect/
│   └── model.pt        # YOLO model for icon detection
└── icon_caption_florence/
    ├── config.json     # Florence2 model configuration
    ├── generation_config.json
    └── model.safetensors # Florence2 model weights
```

## 🔄 Data Flow

### **1. Batch Processing Flow**
```
Shell Script → Python CLI → Omniparser Class → Processing Pipeline
     ↓              ↓              ↓                    ↓
Configuration → Argument Parsing → Model Loading → Image Processing
     ↓              ↓              ↓                    ↓
Error Checking → Validation → OCR Processing → Element Detection
     ↓              ↓              ↓                    ↓
Logging → Execution → Caption Generation → Output Generation
     ↓              ↓              ↓                    ↓
Results → Summary → JSON Files → Annotated Images
```

### **2. Single Image Processing Flow**
```
Input Image → Preprocessing → OCR Processing → Icon Detection
     ↓              ↓              ↓              ↓
PIL Image → Resize/Normalize → Text Extraction → YOLO Inference
     ↓              ↓              ↓              ↓
Validation → Bounding Boxes → Overlap Removal → Caption Generation
     ↓              ↓              ↓              ↓
Error Handling → Element Classification → Visualization → Output Files
```

### **3. OCR Processing Flow**
```
Image Input → Engine Selection → Text Extraction → Result Validation
     ↓              ↓              ↓              ↓
Preprocessing → EasyOCR (Primary) → Confidence Check → Success?
     ↓              ↓              ↓              ↓
Format Check → PaddleOCR (Secondary) → Confidence Check → Success?
     ↓              ↓              ↓              ↓
Output Format → Tesseract (Tertiary) → Confidence Check → Final Result
```

## ⚙️ Configuration Management

### **Configuration Hierarchy**
1. **Shell Script Variables**: User-friendly configuration
2. **Command Line Arguments**: Programmatic configuration
3. **Default Values**: Fallback configuration
4. **Model Configuration**: Model-specific parameters

### **Key Configuration Parameters**
```python
# Detection Parameters
BOX_THRESHOLD = 0.05      # Confidence threshold for detection
IOU_THRESHOLD = 0.05      # IoU threshold for NMS
IMG_SIZE = 3000           # YOLO input size

# OCR Parameters
USE_PADDLEOCR = False     # OCR engine selection
TEXT_THRESHOLD = 0.9      # OCR confidence threshold

# Processing Parameters
BATCH_SIZE = 128          # Caption model batch size
USE_LOCAL_SEMANTICS = True # Enable caption generation
```

## 🛡️ Error Handling

### **Error Handling Strategy**
1. **Input Validation**: Check file existence and format
2. **Model Loading**: Graceful handling of missing models
3. **OCR Fallback**: Automatic engine switching on failure
4. **Processing Recovery**: Continue batch processing on individual failures
5. **Output Validation**: Verify generated files and metadata

### **Error Recovery Mechanisms**
- **OCR Engine Fallback**: Automatic switching between engines
- **Batch Processing**: Continue processing other images on failure
- **Partial Results**: Save partial results when possible
- **Detailed Logging**: Comprehensive error reporting

## ⚡ Performance Considerations

### **Optimization Strategies**
1. **Model Loading**: Load models once and reuse
2. **Batch Processing**: Process multiple images efficiently
3. **Memory Management**: Proper cleanup of large images
4. **Parallel Processing**: Potential for multi-threading

### **Performance Bottlenecks**
- **YOLO Inference**: Most computationally intensive
- **Caption Generation**: Large language model inference
- **OCR Processing**: Text extraction and recognition
- **Image I/O**: File reading and writing

### **Memory Usage**
- **Model Loading**: ~2-4GB for YOLO + caption models
- **Image Processing**: Scales with image size and batch size
- **OCR Processing**: Varies by engine and image complexity

## 🔮 Future Enhancements

### **Planned Improvements**
1. **Advanced Preprocessing**: Image enhancement for better OCR
2. **Multi-threading**: Parallel processing for batch operations
3. **Model Optimization**: Quantization and optimization
4. **Web Interface**: Gradio-based web UI
5. **API Integration**: RESTful API for remote processing

### **Architecture Extensibility**
- **Modular Design**: Easy addition of new OCR engines
- **Plugin System**: Extensible processing pipeline
- **Configuration Management**: Dynamic configuration loading
- **Model Versioning**: Support for multiple model versions

## 📊 Monitoring and Logging

### **Logging Strategy**
- **Shell Script**: Basic progress and error logging
- **Python Script**: Detailed processing logs
- **OCR Engine**: Performance and fallback logging
- **Model Inference**: Timing and accuracy metrics

### **Metrics Collection**
- **Processing Time**: Per-image and batch timing
- **Success Rates**: OCR and detection accuracy
- **Resource Usage**: Memory and CPU utilization
- **Error Rates**: Failure analysis and recovery

This design document provides a comprehensive overview of the OmniParser project architecture, highlighting the modular design, robust error handling, and extensible framework for GUI element detection and text extraction. 