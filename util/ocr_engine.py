"""
OCR Engine with Three-Tier Fallback System

This module provides a robust OCR solution with automatic fallback:
1. EasyOCR (Primary) - Best balance of accuracy and ease of use
2. PaddleOCR (Secondary) - High accuracy, good for complex layouts
3. Tesseract (Tertiary) - Reliable fallback for clean documents

Usage:
    ocr_engine = OCREngine()
    text, bboxes = ocr_engine.extract_text(image_path)
"""

import logging
import time
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR Engine with three-tier fallback system
    """
    
    def __init__(self, 
                 primary_engine: str = 'easyocr',
                 enable_fallbacks: bool = True,
                 timeout_seconds: int = 30,
                 verbose: bool = False):
        """
        Initialize OCR Engine
        
        Args:
            primary_engine: Primary OCR engine ('easyocr', 'paddleocr', 'tesseract')
            enable_fallbacks: Whether to enable fallback engines
            timeout_seconds: Timeout for each OCR engine
            verbose: Whether to print detailed logs
        """
        self.primary_engine = primary_engine
        self.enable_fallbacks = enable_fallbacks
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        
        # Initialize engines
        self.engines = {}
        self._initialize_engines()
        
        # Engine order for fallback
        self.engine_order = ['easyocr', 'paddleocr', 'tesseract']
        
        # Move primary engine to front
        if primary_engine in self.engine_order:
            self.engine_order.remove(primary_engine)
            self.engine_order.insert(0, primary_engine)
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        
        # Initialize EasyOCR
        try:
            import easyocr
            self.engines['easyocr'] = {
                'module': easyocr,
                'reader': easyocr.Reader(['en']),
                'available': True,
                'name': 'EasyOCR'
            }
            if self.verbose:
                logger.info("✓ EasyOCR initialized successfully")
        except ImportError as e:
            logger.warning(f"EasyOCR not available: {e}")
            self.engines['easyocr'] = {'available': False, 'name': 'EasyOCR'}
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            self.engines['easyocr'] = {'available': False, 'name': 'EasyOCR'}
        
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.engines['paddleocr'] = {
                'module': PaddleOCR,
                'ocr': PaddleOCR(use_angle_cls=False, lang='en'),
                'available': True,
                'name': 'PaddleOCR'
            }
            if self.verbose:
                logger.info("✓ PaddleOCR initialized successfully")
        except ImportError as e:
            logger.warning(f"PaddleOCR not available: {e}")
            self.engines['paddleocr'] = {'available': False, 'name': 'PaddleOCR'}
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.engines['paddleocr'] = {'available': False, 'name': 'PaddleOCR'}
        
        # Initialize Tesseract
        try:
            import pytesseract
            # Check if tesseract is installed
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.engines['tesseract'] = {
                    'module': pytesseract,
                    'available': True,
                    'name': 'Tesseract'
                }
                if self.verbose:
                    logger.info("✓ Tesseract initialized successfully")
            else:
                raise Exception("Tesseract not found in PATH")
        except ImportError as e:
            logger.warning(f"pytesseract not available: {e}")
            self.engines['tesseract'] = {'available': False, 'name': 'Tesseract'}
        except Exception as e:
            logger.warning(f"Failed to initialize Tesseract: {e}")
            self.engines['tesseract'] = {'available': False, 'name': 'Tesseract'}
    
    def _load_image(self, image_source: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Load and preprocess image"""
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        elif isinstance(image_source, np.ndarray):
            return image_source
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    
    def _extract_with_easyocr(self, image: np.ndarray, **kwargs) -> Tuple[List[str], List[List[int]]]:
        """Extract text using EasyOCR"""
        if not self.engines['easyocr']['available']:
            raise Exception("EasyOCR not available")
        
        # Default EasyOCR parameters
        easyocr_args = {
            'paragraph': False,
            'text_threshold': 0.9,
            'link_threshold': 0.4,
            'low_text': 0.4,
            'canvas_size': 2560,
            'mag_ratio': 1.5,
            'slope_ths': 0.2,
            'ycenter_ths': 0.5,
            'height_ths': 0.5,
            'width_ths': 0.5,
            'add_margin': 0.1,
            'output_format': 'standard'
        }
        easyocr_args.update(kwargs)
        
        result = self.engines['easyocr']['reader'].readtext(image, **easyocr_args)
        
        texts = []
        bboxes = []
        
        for detection in result:
            bbox, text, confidence = detection
            if confidence > easyocr_args['text_threshold']:
                texts.append(text)
                # Convert bbox to xyxy format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                bboxes.append([int(min(x_coords)), int(min(y_coords)), 
                             int(max(x_coords)), int(max(y_coords))])
        
        return texts, bboxes
    
    def _extract_with_paddleocr(self, image: np.ndarray, **kwargs) -> Tuple[List[str], List[List[int]]]:
        """Extract text using PaddleOCR"""
        if not self.engines['paddleocr']['available']:
            raise Exception("PaddleOCR not available")
        
        # Default PaddleOCR parameters
        text_threshold = kwargs.get('text_threshold', 0.5)
        
        result = self.engines['paddleocr']['ocr'].ocr(image, cls=False)
        
        texts = []
        bboxes = []
        
        if result and result[0]:
            for detection in result[0]:
                bbox, (text, confidence) = detection
                if confidence > text_threshold:
                    texts.append(text)
                    # Convert bbox to xyxy format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bboxes.append([int(min(x_coords)), int(min(y_coords)), 
                                 int(max(x_coords)), int(max(y_coords))])
        
        return texts, bboxes
    
    def _extract_with_tesseract(self, image: np.ndarray, **kwargs) -> Tuple[List[str], List[List[int]]]:
        """Extract text using Tesseract"""
        if not self.engines['tesseract']['available']:
            raise Exception("Tesseract not available")
        
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(image)
        
        # Get text and bounding boxes
        data = self.engines['tesseract']['module'].image_to_data(
            pil_image, 
            output_type=self.engines['tesseract']['module'].Output.DICT,
            config='--psm 6'  # Assume uniform block of text
        )
        
        texts = []
        bboxes = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = data['conf'][i]
            
            # Filter out low confidence and empty text
            if conf > 30 and text:  # Tesseract confidence is 0-100
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                texts.append(text)
                bboxes.append([x, y, x + w, y + h])
        
        return texts, bboxes
    
    def extract_text(self, 
                    image_source: Union[str, Path, Image.Image, np.ndarray],
                    output_format: str = 'xyxy',
                    **kwargs) -> Tuple[List[str], List[List[int]], Dict[str, Any]]:
        """
        Extract text from image with automatic fallback
        
        Args:
            image_source: Image path, PIL Image, or numpy array
            output_format: Bounding box format ('xyxy', 'xywh')
            **kwargs: Additional parameters for OCR engines
            
        Returns:
            Tuple of (texts, bboxes, metadata)
        """
        image = self._load_image(image_source)
        
        # Try each engine in order
        for engine_name in self.engine_order:
            if not self.engines[engine_name]['available']:
                continue
            
            try:
                start_time = time.time()
                
                if engine_name == 'easyocr':
                    texts, bboxes = self._extract_with_easyocr(image, **kwargs)
                elif engine_name == 'paddleocr':
                    texts, bboxes = self._extract_with_paddleocr(image, **kwargs)
                elif engine_name == 'tesseract':
                    texts, bboxes = self._extract_with_tesseract(image, **kwargs)
                else:
                    continue
                
                processing_time = time.time() - start_time
                
                # Check if we got results
                if texts and bboxes:
                    if self.verbose:
                        logger.info(f"✓ {self.engines[engine_name]['name']} succeeded: "
                                  f"{len(texts)} texts in {processing_time:.2f}s")
                    
                    # Convert bbox format if needed
                    if output_format == 'xywh':
                        bboxes = self._convert_xyxy_to_xywh(bboxes)
                    
                    metadata = {
                        'engine_used': engine_name,
                        'engine_name': self.engines[engine_name]['name'],
                        'processing_time': processing_time,
                        'text_count': len(texts),
                        'bbox_count': len(bboxes),
                        'fallback_used': engine_name != self.primary_engine
                    }
                    
                    return texts, bboxes, metadata
                
            except Exception as e:
                if self.verbose:
                    logger.warning(f"✗ {self.engines[engine_name]['name']} failed: {str(e)}")
                continue
        
        # If all engines failed
        raise Exception("All OCR engines failed to extract text")
    
    def _convert_xyxy_to_xywh(self, bboxes: List[List[int]]) -> List[List[int]]:
        """Convert bounding boxes from xyxy to xywh format"""
        converted = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            converted.append([x1, y1, w, h])
        return converted
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        return [name for name, engine in self.engines.items() if engine['available']]
    
    def test_engines(self, test_image_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Test all available OCR engines"""
        results = {}
        
        # Find a test image
        if test_image_path is None:
            test_image_path = self._find_test_image()
        
        if test_image_path is None:
            return {"error": "No test image found"}
        
        for engine_name in self.engine_order:
            if not self.engines[engine_name]['available']:
                results[engine_name] = {"status": "not_available"}
                continue
            
            try:
                start_time = time.time()
                texts, bboxes, metadata = self.extract_text(test_image_path)
                processing_time = time.time() - start_time
                
                results[engine_name] = {
                    "status": "success",
                    "text_count": len(texts),
                    "bbox_count": len(bboxes),
                    "processing_time": processing_time,
                    "sample_texts": texts[:3] if texts else []
                }
                
            except Exception as e:
                results[engine_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results
    
    def _find_test_image(self) -> Optional[Path]:
        """Find a test image in common locations"""
        test_locations = [
            "./imgs/test.png",
            "./imgs/example.png", 
            "./test.png",
            "./example.png",
            "./sample.png"
        ]
        
        for loc in test_locations:
            path = Path(loc)
            if path.exists():
                return path
        
        return None


# Convenience function for backward compatibility
def check_ocr_box(image_source: Union[str, Image.Image], 
                 display_img: bool = True, 
                 output_bb_format: str = 'xywh', 
                 goal_filtering=None, 
                 easyocr_args=None, 
                 use_paddleocr: bool = False) -> Tuple[Tuple[List[str], List[List[int]]], Any]:
    """
    Backward compatibility function for existing code
    
    Args:
        image_source: Image path or PIL Image
        display_img: Whether to display image (not implemented in this version)
        output_bb_format: Bounding box format ('xyxy', 'xywh')
        goal_filtering: Not used, kept for compatibility
        easyocr_args: EasyOCR arguments (passed to OCR engine)
        use_paddleocr: Whether to prefer PaddleOCR (deprecated, use OCREngine instead)
    
    Returns:
        Tuple of (text, bboxes) and goal_filtering
    """
    # Create OCR engine with appropriate primary engine
    primary_engine = 'paddleocr' if use_paddleocr else 'easyocr'
    ocr_engine = OCREngine(primary_engine=primary_engine, enable_fallbacks=True)
    
    try:
        texts, bboxes, metadata = ocr_engine.extract_text(
            image_source, 
            output_format=output_bb_format,
            **(easyocr_args or {})
        )
        return (texts, bboxes), goal_filtering
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ([], []), goal_filtering


# Global OCR engine instance for convenience
_global_ocr_engine = None

def get_global_ocr_engine() -> OCREngine:
    """Get or create a global OCR engine instance"""
    global _global_ocr_engine
    if _global_ocr_engine is None:
        _global_ocr_engine = OCREngine(primary_engine='easyocr', enable_fallbacks=True)
    return _global_ocr_engine 