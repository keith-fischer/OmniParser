#!/usr/bin/env python3
"""
Robust Batch OmniParser Processing Script

This script processes multiple PNG images with enhanced error handling based on
troubleshooting findings. It automatically falls back to EasyOCR when PaddleOCR fails.
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

# Import the modularized Omniparser
from util.omniparser import Omniparser


def get_datetime_prefix():
    """Get current datetime as YYYYMMDD_HHMMSS string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_omniparser_config(
    box_threshold: float = 0.15,
    iou_threshold: float = 0.15,
    use_paddleocr: bool = True,
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Create configuration dictionary for Omniparser
    
    Args:
        box_threshold: Confidence threshold for bounding box detection
        iou_threshold: IoU threshold for non-maximum suppression
        use_paddleocr: Whether to use PaddleOCR instead of EasyOCR
        imgsz: Input image size for YOLO model
        
    Returns:
        Configuration dictionary
    """
    return {
        'som_model_path': 'weights/icon_detect/model.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': box_threshold,
        'iou_threshold': iou_threshold,
        'use_paddleocr': use_paddleocr,
        'imgsz': imgsz,
        'use_local_semantics': True,
        'scale_img': False,
        'batch_size': 128
    }


def process_single_image_with_fallback(
    omniparser_paddleocr: Omniparser,
    omniparser_easyocr: Omniparser,
    image_path: Path,
    output_dir: Path,
    datetime_prefix: str,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Process a single image with automatic fallback from PaddleOCR to EasyOCR
    
    Args:
        omniparser_paddleocr: Omniparser instance configured for PaddleOCR
        omniparser_easyocr: Omniparser instance configured for EasyOCR
        image_path: Path to the image file
        output_dir: Output directory
        datetime_prefix: Datetime prefix for filenames
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing processing results
    """
    print(f"Processing: {image_path.name}")
    
    # Try PaddleOCR first
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}: Using PaddleOCR...")
            start_time = time.time()
            
            result = omniparser_paddleocr.process_single_image(
                image_path=image_path,
                output_dir=output_dir,
                datetime_prefix=f"{datetime_prefix}_paddleocr",
                save_outputs=True
            )
            
            processing_time = time.time() - start_time
            result["ocr_engine"] = "paddleocr"
            result["processing_time"] = processing_time
            result["attempts"] = attempt + 1
            
            print(f"  ✓ PaddleOCR successful: {result['num_elements']} elements in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Check if it's a PaddleOCR-specific error
            if "allocator" in error_msg.lower() or "matmul" in error_msg.lower():
                print(f"  ✗ PaddleOCR failed (attempt {attempt + 1}): {error_type}")
                if attempt < max_retries - 1:
                    print(f"    Retrying with PaddleOCR...")
                    continue
                else:
                    print(f"    Falling back to EasyOCR...")
                    break
            else:
                # Non-PaddleOCR error, don't retry
                print(f"  ✗ Non-PaddleOCR error: {error_type}")
                break
    
    # Fallback to EasyOCR
    try:
        print(f"  Using EasyOCR fallback...")
        start_time = time.time()
        
        result = omniparser_easyocr.process_single_image(
            image_path=image_path,
            output_dir=output_dir,
            datetime_prefix=f"{datetime_prefix}_easyocr",
            save_outputs=True
        )
        
        processing_time = time.time() - start_time
        result["ocr_engine"] = "easyocr"
        result["processing_time"] = processing_time
        result["attempts"] = max_retries + 1
        result["fallback_used"] = True
        result["paddleocr_error"] = error_msg if 'error_msg' in locals() else None
        
        print(f"  ✓ EasyOCR successful: {result['num_elements']} elements in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        # Both OCR engines failed
        error_info = {
            "image_name": image_path.stem,
            "ocr_engine": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "attempts": max_retries + 1,
            "fallback_used": True,
            "paddleocr_error": error_msg if 'error_msg' in locals() else None,
            "easyocr_error": str(e),
            "num_elements": 0,
            "num_ocr_text": 0
        }
        
        print(f"  ✗ Both OCR engines failed: {type(e).__name__}")
        return error_info


def process_batch_robust(
    input_dir: Path,
    img_filter: str,
    output_dir: Path,
    datetime_prefix: str,
    box_threshold: float = 0.10,
    iou_threshold: float = 0.10,
    imgsz: int = 1000,
    prefer_easyocr: bool = False
) -> List[Dict[str, Any]]:
    """
    Process all PNG images with robust error handling and OCR fallback
    
    Args:
        input_dir: Directory containing input images
        img_filter: File filter pattern (e.g., "*.png")
        output_dir: Directory to save outputs
        datetime_prefix: Prefix for output filenames
        box_threshold: Confidence threshold for bounding box detection
        iou_threshold: IoU threshold for non-maximum suppression
        imgsz: Input image size for YOLO model
        prefer_easyocr: If True, use EasyOCR by default instead of PaddleOCR
        
    Returns:
        List of processing results for each image
    """
    # Find all PNG files
    png_files = list(input_dir.glob(img_filter))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return []
    
    print(f"Found {len(png_files)} PNG files to process")
    print(f"Using datetime prefix: {datetime_prefix}")
    print(f"Prefer EasyOCR: {prefer_easyocr}")
    
    # Create Omniparser instances for both OCR engines
    config_paddleocr = create_omniparser_config(
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=True,
        imgsz=imgsz
    )
    
    config_easyocr = create_omniparser_config(
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=False,  # Use EasyOCR
        imgsz=imgsz
    )
    
    print("Initializing Omniparser instances...")
    omniparser_paddleocr = Omniparser(config_paddleocr)
    omniparser_easyocr = Omniparser(config_easyocr)
    
    # Process each image
    all_results = []
    paddleocr_successes = 0
    easyocr_fallbacks = 0
    total_failures = 0
    
    for i, png_file in enumerate(png_files, 1):
        print(f"\n[{i}/{len(png_files)}] Processing: {png_file.name}")
        
        try:
            if prefer_easyocr:
                # Use EasyOCR by default
                result = omniparser_easyocr.process_single_image(
                    image_path=png_file,
                    output_dir=output_dir,
                    datetime_prefix=f"{datetime_prefix}_easyocr",
                    save_outputs=True
                )
                result["ocr_engine"] = "easyocr"
                result["preferred_engine"] = True
                easyocr_fallbacks += 1
            else:
                # Try PaddleOCR with fallback to EasyOCR
                result = process_single_image_with_fallback(
                    omniparser_paddleocr,
                    omniparser_easyocr,
                    png_file,
                    output_dir,
                    datetime_prefix
                )
                
                if result.get("ocr_engine") == "paddleocr":
                    paddleocr_successes += 1
                elif result.get("ocr_engine") == "easyocr":
                    easyocr_fallbacks += 1
                else:
                    total_failures += 1
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Unexpected error processing {png_file.name}: {str(e)}")
            all_results.append({
                "image_name": png_file.stem,
                "ocr_engine": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "num_elements": 0,
                "num_ocr_text": 0
            })
            total_failures += 1
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(png_files)}")
    print(f"PaddleOCR successes: {paddleocr_successes}")
    print(f"EasyOCR fallbacks: {easyocr_fallbacks}")
    print(f"Total failures: {total_failures}")
    print(f"Success rate: {((paddleocr_successes + easyocr_fallbacks) / len(png_files) * 100):.1f}%")
    
    return all_results


def save_batch_summary(results: List[Dict[str, Any]], output_dir: Path, datetime_prefix: str):
    """Save a summary of batch processing results"""
    summary_path = output_dir / f"{datetime_prefix}_robust_batch_summary.json"
    
    # Calculate statistics
    total_images = len(results)
    successful_processing = len([r for r in results if r.get("ocr_engine") in ["paddleocr", "easyocr"]])
    failed_processing = len([r for r in results if r.get("ocr_engine") == "failed"])
    paddleocr_successes = len([r for r in results if r.get("ocr_engine") == "paddleocr"])
    easyocr_fallbacks = len([r for r in results if r.get("ocr_engine") == "easyocr"])
    
    summary_data = {
        "batch_info": {
            "total_images": total_images,
            "successful_processing": successful_processing,
            "failed_processing": failed_processing,
            "paddleocr_successes": paddleocr_successes,
            "easyocr_fallbacks": easyocr_fallbacks,
            "success_rate": successful_processing / total_images if total_images > 0 else 0,
            "datetime_prefix": datetime_prefix
        },
        "results": results
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Robust batch summary saved: {summary_path}")


def run_robust_omniparse(
    img_path: str = "./imgs",
    img_filter: str = "*.png",
    out_dir: str = "./output",
    box: float = 0.05, 
    iou: float = 0.05,
    img_size: int = 3000,
    prefer_easyocr: bool = False
):
    """
    Main function to run robust OmniParser batch processing
    
    Args:
        img_path: Path to input images directory
        img_filter: File filter pattern
        out_dir: Output directory path
        box: Box threshold for detection
        iou: IoU threshold for detection
        img_size: Input image size
        prefer_easyocr: If True, use EasyOCR by default instead of PaddleOCR
    """
    input_dir = Path(img_path)
    output_dir = Path(out_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get datetime prefix for this batch
    datetime_prefix = get_datetime_prefix()
    
    print(f"Starting Robust OmniParser batch processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"File filter: {img_filter}")
    print(f"Box threshold: {box}")
    print(f"IoU threshold: {iou}")
    print(f"Image size: {img_size}")
    print(f"Prefer EasyOCR: {prefer_easyocr}")
    
    # Process the batch with robust error handling
    results = process_batch_robust(
        input_dir=input_dir,
        img_filter=img_filter,
        output_dir=output_dir,
        datetime_prefix=datetime_prefix,
        box_threshold=box,
        iou_threshold=iou,
        imgsz=img_size,
        prefer_easyocr=prefer_easyocr
    )
    
    # Save batch summary
    save_batch_summary(results, output_dir, datetime_prefix)
    
    return results


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(
        description="Robust batch process PNG images with OmniParser (with OCR fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_run_omniparser_robust.py --input_dir ./images --output_dir ./output
  python batch_run_omniparser_robust.py --input_dir ./screenshots --box 0.1 --iou 0.1
  python batch_run_omniparser_robust.py --input_dir ./test --prefer_easyocr
        """
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default="./imgs",
        help="Input directory containing PNG images (default: ./imgs)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--img_filter",
        type=str,
        default="*.png",
        help="File filter pattern (default: *.png)"
    )
    
    parser.add_argument(
        "--box",
        type=float,
        default=0.05,
        help="Box threshold for detection (default: 0.05)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.05,
        help="IoU threshold for detection (default: 0.05)"
    )
    
    parser.add_argument(
        "--img_size",
        type=int,
        default=3000,
        help="Input image size for YOLO model (default: 3000)"
    )
    
    parser.add_argument(
        "--prefer_easyocr",
        action="store_true",
        help="Use EasyOCR by default instead of PaddleOCR (recommended for reliability)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Run the robust batch processing
    try:
        results = run_robust_omniparse(
            img_path=args.input_dir,
            img_filter=args.img_filter,
            out_dir=args.output_dir,
            box=args.box,
            iou=args.iou,
            img_size=args.img_size,
            prefer_easyocr=args.prefer_easyocr
        )
        
        print(f"\nRobust batch processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during robust batch processing: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 