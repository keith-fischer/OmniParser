from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from datetime import datetime


class Omniparser(object):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Omniparser with configuration
        
        Args:
            config: Dictionary containing configuration parameters
                   If None, uses default configuration
        """
        if config is None:
            config = {
                'som_model_path': 'weights/icon_detect/model.pt',
                'caption_model_name': 'florence2',
                'caption_model_path': 'weights/icon_caption_florence',
                'BOX_TRESHOLD': 0.15,
                'iou_threshold': 0.15,
                'use_paddleocr': False,
                'imgsz': 640,
                'use_local_semantics': True,
                'scale_img': False,
                'batch_size': 128
            }
        
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize models
        print("Loading YOLO model...")
        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        
        print("Loading caption model...")
        self.caption_model_processor = get_caption_model_processor(
            model_name=config['caption_model_name'], 
            model_name_or_path=config['caption_model_path'], 
            device=device
        )
        
        print('Omniparser initialized!!!')

    def test_module(self, test_image_path: Optional[Union[str, Path]] = None, save_test_output: bool = True) -> Dict[str, Any]:
        """
        Test the Omniparser module with default parameters and a test image
        
        Args:
            test_image_path: Path to test image. If None, looks for common test images
            save_test_output: Whether to save test outputs to files
            
        Returns:
            Dict containing test results and validation information
        """
        print("=" * 60)
        print("OMNIPARSER MODULE TEST")
        print("=" * 60)
        
        # Find test image
        if test_image_path is None:
            test_image_path = self._find_test_image()
        
        if test_image_path is None:
            return {
                "test_status": "FAILED",
                "error": "No test image found. Please provide a test_image_path or place a test image in common locations.",
                "suggested_locations": [
                    "./imgs/test.png",
                    "./imgs/example.png", 
                    "./test.png",
                    "./example.png"
                ]
            }
        
        test_image_path = Path(test_image_path)
        
        if not test_image_path.exists():
            return {
                "test_status": "FAILED",
                "error": f"Test image not found at: {test_image_path}",
                "test_image_path": str(test_image_path)
            }
        
        print(f"Using test image: {test_image_path}")
        print(f"Configuration: {self.config}")
        
        # Test 1: Basic image loading
        print("\n1. Testing image loading...")
        try:
            test_image = Image.open(test_image_path)
            print(f"   ✓ Image loaded successfully: {test_image.size}")
        except Exception as e:
            return {
                "test_status": "FAILED",
                "error": f"Failed to load test image: {str(e)}",
                "test_image_path": str(test_image_path)
            }
        
        # Test 2: OCR processing
        print("2. Testing OCR processing...")
        try:
            ocr_bbox_rslt, _ = check_ocr_box(
                test_image, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
                use_paddleocr=self.config.get('use_paddleocr', True)
            )
            text, ocr_bbox = ocr_bbox_rslt
            print(f"   ✓ OCR completed: {len(text)} text items detected")
        except Exception as e:
            return {
                "test_status": "FAILED",
                "error": f"OCR processing failed: {str(e)}",
                "test_image_path": str(test_image_path)
            }
        
        # Test 3: Full Omniparser processing
        print("3. Testing full Omniparser processing...")
        try:
            result = self.process_single_image(
                image_path=test_image_path,
                output_dir=Path("./test_output") if save_test_output else None,
                datetime_prefix="test",
                save_outputs=save_test_output
            )
            print(f"   ✓ Full processing completed successfully")
            print(f"   ✓ Detected {result['num_elements']} elements")
            print(f"   ✓ Found {result['num_ocr_text']} OCR text items")
            
        except Exception as e:
            return {
                "test_status": "FAILED",
                "error": f"Full processing failed: {str(e)}",
                "test_image_path": str(test_image_path),
                "ocr_text_count": len(text),
                "ocr_bbox_count": len(ocr_bbox)
            }
        
        # Test 4: Base64 processing
        print("4. Testing base64 processing...")
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            test_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Process base64
            annotated_base64, parsed_content = self.parse_base64(image_base64)
            print(f"   ✓ Base64 processing completed: {len(parsed_content)} elements")
            
        except Exception as e:
            return {
                "test_status": "PARTIAL_SUCCESS",
                "error": f"Base64 processing failed: {str(e)}",
                "test_image_path": str(test_image_path),
                "full_processing_success": True,
                "num_elements": result['num_elements'],
                "num_ocr_text": result['num_ocr_text']
            }
        
        # Test 5: PIL Image processing
        print("5. Testing PIL Image processing...")
        try:
            annotated_pil, parsed_content_pil = self.parse_image(test_image)
            print(f"   ✓ PIL Image processing completed: {len(parsed_content_pil)} elements")
            
        except Exception as e:
            return {
                "test_status": "PARTIAL_SUCCESS",
                "error": f"PIL Image processing failed: {str(e)}",
                "test_image_path": str(test_image_path),
                "full_processing_success": True,
                "base64_processing_success": True,
                "num_elements": result['num_elements'],
                "num_ocr_text": result['num_ocr_text']
            }
        
        # All tests passed
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - OMNIPARSER MODULE IS WORKING CORRECTLY")
        print("=" * 60)
        
        test_results = {
            "test_status": "SUCCESS",
            "test_image_path": str(test_image_path),
            "image_size": test_image.size,
            "configuration": self.config,
            "results": {
                "num_elements": result['num_elements'],
                "num_ocr_text": result['num_ocr_text'],
                "ocr_text_count": len(text),
                "ocr_bbox_count": len(ocr_bbox)
            },
            "processing_methods_tested": [
                "image_loading",
                "ocr_processing", 
                "full_processing",
                "base64_processing",
                "pil_processing"
            ],
            "output_files_saved": save_test_output,
            "test_timestamp": datetime.now().isoformat()
        }
        
        if save_test_output:
            test_results["output_directory"] = str(Path("./test_output"))
        
        return test_results

    def _find_test_image(self) -> Optional[Path]:
        """
        Find a suitable test image in common locations
        
        Returns:
            Path to test image if found, None otherwise
        """
        common_test_locations = [
            "./imgs/test.png",
            "./imgs/example.png",
            "./imgs/sample.png",
            "./test.png",
            "./example.png",
            "./sample.png",
            "./imgs/*.png",  # Any PNG in imgs directory
            "./*.png"        # Any PNG in current directory
        ]
        
        for location in common_test_locations:
            if "*" in location:
                # Handle glob patterns
                import glob
                matches = glob.glob(location)
                if matches:
                    return Path(matches[0])
            else:
                # Handle specific paths
                path = Path(location)
                if path.exists():
                    return path
        
        return None

    def parse_base64(self, image_base64: str) -> Tuple[str, List[Dict]]:
        """
        Parse image from base64 string
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Tuple of (base64_annotated_image, parsed_content_list)
        """
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        return self.parse_image(image)

    def parse_image(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """
        Parse PIL Image object
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (base64_annotated_image, parsed_content_list)
        """
        print('image size:', image.size)
        
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        (text, ocr_bbox), _ = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', 
            easyocr_args={'text_threshold': 0.8}, 
            use_paddleocr=self.config.get('use_paddleocr', False)
        )
        
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, 
            self.som_model, 
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'], 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            use_local_semantics=self.config.get('use_local_semantics', True), 
            iou_threshold=self.config.get('iou_threshold', 0.7), 
            scale_img=self.config.get('scale_img', False), 
            batch_size=self.config.get('batch_size', 128)
        )

        return dino_labeled_img, parsed_content_list

    def process_single_image(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        datetime_prefix: Optional[str] = None,
        save_outputs: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image and optionally save the three output files
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save outputs (if None, outputs won't be saved)
            datetime_prefix: Prefix for output filenames (if None, auto-generated)
            save_outputs: Whether to save output files
            
        Returns:
            Dict containing processing results and file paths
        """
        print(f"Processing: {image_path.name}")
        
        # Generate datetime prefix if not provided
        if datetime_prefix is None:
            datetime_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load and process the image
        image_input = Image.open(image_path)
        
        # Calculate box overlay ratio based on image size
        box_overlay_ratio = image_input.size[0] / 3200
        
        # Define configuration for drawing bounding boxes
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
            "text_padding": 5
        }
        
        # Run OCR to get text regions
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=self.config.get('use_paddleocr', True)
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Process with OmniParser
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input, 
            self.som_model, 
            BOX_TRESHOLD=self.config['BOX_TRESHOLD'], 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            iou_threshold=self.config.get('iou_threshold', 0.15), 
            imgsz=self.config.get('imgsz', 640)
        )
        
        # Prepare result data
        result = {
            "image_name": image_path.stem,
            "image_size": image_input.size,
            "num_elements": len(parsed_content_list),
            "num_ocr_text": len(text),
            "detected_elements": parsed_content_list,
            "label_coordinates": label_coordinates,
            "ocr_text": text,
            "ocr_bbox": ocr_bbox,
            "annotated_image_base64": dino_labeled_img,
            "processing_parameters": {
                "box_threshold": self.config['BOX_TRESHOLD'],
                "iou_threshold": self.config.get('iou_threshold', 0.15),
                "use_paddleocr": self.config.get('use_paddleocr', True),
                "imgsz": self.config.get('imgsz', 640)
            }
        }
        
        # Save outputs if requested
        if save_outputs and output_dir is not None:
            result.update(self._save_outputs(
                image_input, dino_labeled_img, result, 
                image_path, output_dir, datetime_prefix
            ))
        
        return result

    def _save_outputs(
        self, 
        image_input: Image.Image, 
        dino_labeled_img: str, 
        result: Dict[str, Any],
        image_path: Path, 
        output_dir: Path, 
        datetime_prefix: str
    ) -> Dict[str, str]:
        """
        Save the three output files for a processed image
        
        Returns:
            Dict containing file paths
        """
        # Create output subdirectory for this image
        image_name = image_path.stem
        image_output_dir = output_dir / image_name
        image_output_dir.mkdir(exist_ok=True)
        
        # Convert thresholds to integers for filename
        box_threshold_int = int(self.config['BOX_TRESHOLD'] * 100)
        iou_threshold_int = int(self.config.get('iou_threshold', 0.15) * 100)
        
        # 1. Copy of original PNG
        original_copy_path = image_output_dir / f"{datetime_prefix}_{image_name}_original.png"
        image_input.save(original_copy_path)
        
        # 2. Marked up PNG with detected regions
        marked_up_path = image_output_dir / f"{datetime_prefix}_{image_name}_omni_b{box_threshold_int}_i{iou_threshold_int}.png"
        marked_up_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
        marked_up_image.save(marked_up_path)
        
        # 3. JSON file with all detected elements
        json_path = image_output_dir / f"{datetime_prefix}_{image_name}_elements.json"
        
        json_data = {
            "image_info": {
                "original_path": str(image_path),
                "image_size": image_input.size,
                "processing_parameters": result["processing_parameters"]
            },
            "detected_elements": result["detected_elements"],
            "label_coordinates": result["label_coordinates"],
            "ocr_text": result["ocr_text"],
            "ocr_bbox": result["ocr_bbox"]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved: {original_copy_path}")
        print(f"  ✓ Saved: {marked_up_path}")
        print(f"  ✓ Saved: {json_path}")
        
        return {
            "original_copy_path": str(original_copy_path),
            "marked_up_path": str(marked_up_path),
            "json_path": str(json_path)
        }


def test_omniparser_module(test_image_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Standalone function to test the Omniparser module
    
    Args:
        test_image_path: Path to test image. If None, auto-discovers test image
        
    Returns:
        Dict containing test results
    """
    print("Testing Omniparser module...")
    
    try:
        # Initialize with default configuration
        omniparser = Omniparser()
        
        # Run comprehensive test
        test_results = omniparser.test_module(test_image_path=test_image_path)
        
        return test_results
        
    except Exception as e:
        return {
            "test_status": "FAILED",
            "error": f"Module initialization failed: {str(e)}",
            "test_image_path": str(test_image_path) if test_image_path else None
        }


if __name__ == "__main__":
    # Run test when module is executed directly
    test_results = test_omniparser_module()
    print(f"\nTest Results: {test_results['test_status']}")
    
    if test_results['test_status'] != 'SUCCESS':
        print(f"Error: {test_results.get('error', 'Unknown error')}")
    else:
        print("✓ Omniparser module is working correctly!")