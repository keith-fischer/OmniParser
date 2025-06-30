# OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent

<p align="center">
  <img src="imgs/logo.png" alt="Logo">
</p>
<!-- <a href="https://trendshift.io/repositories/12975" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12975" alt="microsoft%2FOmniParser | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a> -->

[![arXiv](https://img.shields.io/badge/Paper-green)](https://arxiv.org/abs/2408.00203)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [[Models V2](https://huggingface.co/microsoft/OmniParser-v2.0)] [[Models V1.5](https://huggingface.co/microsoft/OmniParser)] [[HuggingFace Space Demo](https://huggingface.co/spaces/microsoft/OmniParser-v2)]

**OmniParser** is a comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements, which significantly enhances the ability of GPT-4V to generate actions that can be accurately grounded in the corresponding regions of the interface. 

## News
- [2025/3] We support local logging of trajecotry so that you can use OmniParser+OmniTool to build training data pipeline for your favorate agent in your domain. [Documentation WIP]
- [2025/3] We are gradually adding multi agents orchstration and improving user interface in OmniTool for better experience.
- [2025/2] We release OmniParser V2 [checkpoints](https://huggingface.co/microsoft/OmniParser-v2.0). [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EWXbVESKWo9Buu6OYCwg06wBeoM97C6EOTG6RjvWLEN1Qg?e=alnHGC)
- [2025/2] We introduce OmniTool: Control a Windows 11 VM with OmniParser + your vision model of choice. OmniTool supports out of the box the following large language models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use. [Watch Video](https://1drv.ms/v/c/650b027c18d5a573/EehZ7RzY69ZHn-MeQHrnnR4BCj3by-cLLpUVlxMjF4O65Q?e=8LxMgX)
- [2025/1] V2 is coming. We achieve new state of the art results 39.5% on the new grounding benchmark [Screen Spot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/main) with OmniParser v2 (will be released soon)! Read more details [here](https://github.com/microsoft/OmniParser/tree/master/docs/Evaluation.md).
- [2024/11] We release an updated version, OmniParser V1.5 which features 1) more fine grained/small icon detection, 2) prediction of whether each screen element is interactable or not. Examples in the demo.ipynb. 
- [2024/10] OmniParser was the #1 trending model on huggingface model hub (starting 10/29/2024). 
- [2024/10] Feel free to checkout our demo on [huggingface space](https://huggingface.co/spaces/microsoft/OmniParser)! (stay tuned for OmniParser + Claude Computer Use)
- [2024/10] Both Interactive Region Detection Model and Icon functional description model are released! [Hugginface models](https://huggingface.co/microsoft/OmniParser)
- [2024/09] OmniParser achieves the best performance on [Windows Agent Arena](https://microsoft.github.io/WindowsAgentArena/)! 

## Install 

### Quick Installation (Recommended)

**For Mac/Linux:**
```bash
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
chmod +x install.sh
./install.sh
```

**For Windows:**
```cmd
git clone https://github.com/microsoft/OmniParser.git
cd OmniParser
install.bat
```

### Manual Installation

First clone the repo, and then install environment:

**Option 1: Auto-detection (Recommended)**
```bash
cd OmniParser
python install_dependencies.py
```

**Option 2: Platform-specific requirements**
```bash
cd OmniParser
# For Mac
pip install -r requirements_mac.txt

# For Windows (with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_windows.txt

# For Linux
pip install -r requirements.txt
```

**Option 3: Legacy conda installation**
```bash
cd OmniParser
conda create -n "omni" python==3.12
conda activate omni
pip install -r requirements.txt
```

Ensure you have the V2 weights downloaded in weights folder (ensure caption weights folder is called icon_caption_florence). If not download them with:
```
   # download the model checkpoints to local directory OmniParser/weights/
   for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
   mv weights/icon_caption weights/icon_caption_florence
```

<!-- ## [deprecated]
Then download the model ckpts files in: https://huggingface.co/microsoft/OmniParser, and put them under weights/, default folder structure is: weights/icon_detect, weights/icon_caption_florence, weights/icon_caption_blip2. 

For v1: 
convert the safetensor to .pt file. 
```python
python weights/convert_safetensor_to_pt.py

For v1.5: 
download 'model_v1_5.pt' from https://huggingface.co/microsoft/OmniParser/tree/main/icon_detect_v1_5, make a new dir: weights/icon_detect_v1_5, and put it inside the folder. No weight conversion is needed. 
``` -->

## Examples:
We put together a few simple examples in the demo.ipynb. 

## Debug and Testing
All debugging, testing, demo, and troubleshooting scripts are located in the `/debug` folder. This includes:
- OCR engine testing and comparison scripts
- Gradio demo applications
- Batch processing scripts
- Troubleshooting tools
- Interactive Jupyter notebooks

See [debug/README.md](debug/README.md) for comprehensive documentation of all testing scripts and development history.

## Gradio Demo
To run gradio demo, simply run:
```python
python gradio_demo.py
```

## OCR Engine and Fallback System

OmniParser includes a robust three-tier OCR fallback system:

### **Available OCR Engines:**
1. **EasyOCR** (Primary) - Best balance of accuracy and ease of use
2. **PaddleOCR** (Secondary) - High accuracy, good for complex layouts  
3. **Tesseract** (Tertiary) - Reliable fallback for clean documents

### **Testing OCR Engines:**
```bash
python test_ocr_fallback.py
```

This will test all available engines and provide a comparison table of word counts and element counts.

#### **Example Test Results:**

**Test Image:** `imgs/tsserrors/frame_0034.png`

**OCR Engine Comparison Table:**

| Metric      |   easyocr   |  paddleocr  |  tesseract  |
|-------------|:-----------:|:-----------:|:-----------:|
| Word Count  |     54      |     54      |     54      |
| Elem Count  |     33      |     33      |     33      |

**Performance Results:**
- **EasyOCR**: 1.90s processing time
- **PaddleOCR**: 1.01s processing time (fastest)
- **Tesseract**: 1.00s processing time (fastest)

**Key Observations:**
- All three engines detected exactly the same number of text elements (33) and total words (54)
- Sample texts detected: "UCSan", "Maintain object", "Menu"
- Fallback system successfully used Tesseract when EasyOCR and PaddleOCR failed in fallback test
- Results saved to `ocr_easyocr.json`, `ocr_paddleocr.json`, and `ocr_tesseract.json` for detailed inspection

## Advanced OCR Improvements

For challenging GUI/business form images with difficult text conditions (light on dark backgrounds, low contrast, colored text), additional preprocessing can significantly improve OCR accuracy.

### **Current OCR Engine Capabilities**

**EasyOCR Built-in Processing:**
- Automatic image resizing to optimal size
- Basic normalization and RGB conversion
- Multi-scale text detection
- CRAFT text detection + CRNN recognition

**Limitations:**
- Generic optimization, not specific to GUI forms
- Fixed preprocessing for all images
- Limited handling of extreme contrast cases
- No advanced color space analysis

### **Recommended Preprocessing Techniques**

#### **1. Multi-Scale Processing**
- **Upscaling**: 2x-4x resolution increase (not just 2-3x)
- **Rationale**: OCR engines work better with higher resolution, especially for small UI text
- **Implementation**: Use Lanczos or cubic interpolation for best quality

#### **2. Advanced Contrast Enhancement**
- **Adaptive Histogram Equalization (CLAHE)**: Better than simple histogram equalization
- **Multi-channel processing**: Process RGB channels separately, then combine
- **Local contrast enhancement**: Apply contrast enhancement in sliding windows
- **Gamma correction**: Adjust gamma based on image brightness distribution

#### **3. Color Space Transformations**
- **HSV/LAB conversion**: Better for color-based text detection
- **Channel separation**: Extract the channel with highest text-background contrast
- **Color clustering**: Use K-means to identify text vs background colors

#### **4. Advanced Thresholding Techniques**
- **Otsu's method**: Automatic threshold selection
- **Adaptive thresholding**: Different thresholds for different image regions
- **Multi-thresholding**: Apply multiple thresholds and combine results
- **Edge-based thresholding**: Use Canny edges to guide threshold selection

#### **5. Sharpening and Noise Reduction**
- **Unsharp masking**: More controlled than simple sharpening
- **Bilateral filtering**: Reduce noise while preserving edges
- **Morphological operations**: Clean up text regions

#### **6. Text-Specific Enhancements**
- **Stroke width transform**: Identify text-like regions
- **MSER (Maximally Stable Extremal Regions)**: Detect text regions
- **Connected component analysis**: Group related text elements

### **Implementation Strategy**

#### **Phase 1: Basic Enhancement**
1. Upscale image 2-4x
2. Convert to grayscale using luminance formula
3. Apply CLAHE
4. Add to OCR engine as preprocessing step

#### **Phase 2: Advanced Processing**
1. Multi-channel analysis
2. Adaptive thresholding
3. Morphological operations
4. Edge-preserving smoothing

#### **Phase 3: Intelligent Selection**
1. Try multiple preprocessing approaches
2. Use OCR confidence scores to select best result
3. Combine results from different preprocessing methods

### **Specific Techniques for GUI/Business Forms**

#### **For Light Text on Dark Background:**
- Invert image first
- Use high-pass filtering
- Apply aggressive contrast enhancement

#### **For Low Contrast Text:**
- CLAHE with large window size
- Multi-scale contrast enhancement
- Edge enhancement before OCR

#### **For Colored Text:**
- Channel separation (R, G, B, H, S, V)
- Color clustering to identify text colors
- Adaptive thresholding per color channel

### **Recommended Preprocessing Pipeline**

```python
def preprocess_for_ocr(image):
    # 1. Upscale (2-4x)
    # 2. Convert to grayscale using best channel
    # 3. Apply CLAHE
    # 4. Adaptive thresholding
    # 5. Morphological cleanup
    # 6. Sharpening
    pass
```

### **Hybrid Approach**

```python
def extract_text_hybrid(image):
    # Try EasyOCR on original image
    result_original = easyocr.extract_text(image)
    
    # Apply preprocessing
    preprocessed = preprocess_image(image)
    result_preprocessed = easyocr.extract_text(preprocessed)
    
    # Compare confidence scores and return best result
    return best_result(result_original, result_preprocessed)
```

### **Performance Considerations**

- **Memory usage**: Upscaling 4x increases memory 16x
- **Processing time**: Preprocessing adds 0.5-2s per image
- **Quality vs speed tradeoff**: More preprocessing = better accuracy but slower

### **Recommended Libraries**

- **OpenCV**: Core image processing
- **scikit-image**: Advanced morphological operations
- **PIL/Pillow**: Basic image operations
- **numpy**: Array operations for custom algorithms

### **When Additional Preprocessing Helps**

**EasyOCR + Preprocessing is Beneficial When:**
1. **Extreme contrast issues**: Very light text on dark backgrounds
2. **Color-specific problems**: Text in specific colors that blend with background
3. **Noise and artifacts**: Screenshots with compression artifacts
4. **Small text**: UI elements that are too small even after EasyOCR's scaling
5. **Mixed content**: Images with both text and graphics

**Real-World Examples:**
- **Dark mode applications**: White text on dark backgrounds
- **Low contrast forms**: Gray text on light gray backgrounds
- **Colored UI elements**: Blue text on blue-tinted backgrounds
- **Compressed screenshots**: JPEG artifacts affecting text clarity

## Model Weights License
For the model checkpoints on huggingface model hub, please note that icon_detect model is under AGPL license since it is a license inherited from the original yolo model. And icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model: https://huggingface.co/microsoft/OmniParser.

## 📚 Citation
Our technical report can be found [here](https://arxiv.org/abs/2408.00203).
If you find our work useful, please consider citing our work:
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
