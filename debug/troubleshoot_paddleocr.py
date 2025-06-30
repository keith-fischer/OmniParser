#!/usr/bin/env python3
"""
PaddleOCR Troubleshooting Script

This script analyzes failed images to identify and troubleshoot PaddleOCR errors.
It provides detailed error information and tries different configurations.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List
import time

# Import Omniparser and utilities
from util.omniparser import Omniparser
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor
from PIL import Image
import torch


def analyze_image_properties(image_path: Path) -> Dict[str, Any]:
    """Analyze basic image properties that might cause PaddleOCR issues"""
    try:
        image = Image.open(image_path)
        
        properties = {
            "file_size_mb": image_path.stat().st_size / (1024 * 1024),
            "image_size": image.size,
            "image_mode": image.mode,
            "image_format": image.format,
            "aspect_ratio": image.size[0] / image.size[1] if image.size[1] > 0 else 0,
            "total_pixels": image.size[0] * image.size[1],
            "memory_usage_mb": (image.size[0] * image.size[1] * len(image.getbands())) / (1024 * 1024)
        }
        
        return properties
        
    except Exception as e:
        return {"error": f"Failed to analyze image: {str(e)}"}


def test_ocr_with_different_configs(image_path: Path) -> Dict[str, Any]:
    """Test OCR with different configurations to identify the issue"""
    results = {}
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"Testing image: {image_path.name} ({image.size})")
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}
    
    # Test 1: PaddleOCR with default settings
    print("  1. Testing PaddleOCR (default)...")
    try:
        start_time = time.time()
        ocr_result, _ = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_result
        processing_time = time.time() - start_time
        
        results["paddleocr_default"] = {
            "status": "SUCCESS",
            "processing_time": processing_time,
            "text_count": len(text),
            "bbox_count": len(ocr_bbox),
            "text_samples": text[:5] if text else []  # First 5 text items
        }
        print(f"     ✓ Success: {len(text)} text items in {processing_time:.2f}s")
        
    except Exception as e:
        error_info = {
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        results["paddleocr_default"] = error_info
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)}")
    
    # Test 2: EasyOCR instead of PaddleOCR
    print("  2. Testing EasyOCR...")
    try:
        start_time = time.time()
        ocr_result, _ = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=False
        )
        text, ocr_bbox = ocr_result
        processing_time = time.time() - start_time
        
        results["easyocr"] = {
            "status": "SUCCESS",
            "processing_time": processing_time,
            "text_count": len(text),
            "bbox_count": len(ocr_bbox),
            "text_samples": text[:5] if text else []
        }
        print(f"     ✓ Success: {len(text)} text items in {processing_time:.2f}s")
        
    except Exception as e:
        error_info = {
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        results["easyocr"] = error_info
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)}")
    
    # Test 3: PaddleOCR with different text threshold
    print("  3. Testing PaddleOCR (lower threshold)...")
    try:
        start_time = time.time()
        ocr_result, _ = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.5}, 
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_result
        processing_time = time.time() - start_time
        
        results["paddleocr_lower_threshold"] = {
            "status": "SUCCESS",
            "processing_time": processing_time,
            "text_count": len(text),
            "bbox_count": len(ocr_bbox),
            "text_samples": text[:5] if text else []
        }
        print(f"     ✓ Success: {len(text)} text items in {processing_time:.2f}s")
        
    except Exception as e:
        error_info = {
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        results["paddleocr_lower_threshold"] = error_info
        print(f"     ✗ Failed: {type(e).__name__}: {str(e)}")
    
    # Test 4: Resized image (if original is very large)
    if image.size[0] > 3000 or image.size[1] > 3000:
        print("  4. Testing with resized image...")
        try:
            # Resize to reasonable size
            resized_image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
            start_time = time.time()
            ocr_result, _ = check_ocr_box(
                resized_image, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
                use_paddleocr=True
            )
            text, ocr_bbox = ocr_result
            processing_time = time.time() - start_time
            
            results["paddleocr_resized"] = {
                "status": "SUCCESS",
                "processing_time": processing_time,
                "text_count": len(text),
                "bbox_count": len(ocr_bbox),
                "text_samples": text[:5] if text else [],
                "resized_to": resized_image.size
            }
            print(f"     ✓ Success: {len(text)} text items in {processing_time:.2f}s")
            
        except Exception as e:
            error_info = {
                "status": "FAILED",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            results["paddleocr_resized"] = error_info
            print(f"     ✗ Failed: {type(e).__name__}: {str(e)}")
    
    return results


def test_full_omniparser_processing(image_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Test full Omniparser processing with different OCR configurations"""
    results = {}
    
    # Test 1: Default configuration (PaddleOCR)
    print(f"\nTesting full Omniparser processing with PaddleOCR...")
    try:
        config_paddleocr = {
            'som_model_path': 'weights/icon_detect/model.pt',
            'caption_model_name': 'florence2',
            'caption_model_path': 'weights/icon_caption_florence',
            'BOX_TRESHOLD': 0.15,
            'iou_threshold': 0.15,
            'use_paddleocr': True,
            'imgsz': 640,
            'use_local_semantics': True,
            'scale_img': False,
            'batch_size': 128
        }
        
        omniparser = Omniparser(config_paddleocr)
        start_time = time.time()
        
        result = omniparser.process_single_image(
            image_path=image_path,
            output_dir=output_dir,
            datetime_prefix="troubleshoot_paddleocr",
            save_outputs=True
        )
        
        processing_time = time.time() - start_time
        
        results["full_processing_paddleocr"] = {
            "status": "SUCCESS",
            "processing_time": processing_time,
            "num_elements": result['num_elements'],
            "num_ocr_text": result['num_ocr_text'],
            "output_files": {
                "original_copy": result.get('original_copy_path'),
                "marked_up": result.get('marked_up_path'),
                "json": result.get('json_path')
            }
        }
        print(f"  ✓ Full processing with PaddleOCR successful: {result['num_elements']} elements in {processing_time:.2f}s")
        
    except Exception as e:
        error_info = {
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        results["full_processing_paddleocr"] = error_info
        print(f"  ✗ Full processing with PaddleOCR failed: {type(e).__name__}: {str(e)}")
    
    # Test 2: EasyOCR configuration
    print(f"Testing full Omniparser processing with EasyOCR...")
    try:
        config_easyocr = {
            'som_model_path': 'weights/icon_detect/model.pt',
            'caption_model_name': 'florence2',
            'caption_model_path': 'weights/icon_caption_florence',
            'BOX_TRESHOLD': 0.15,
            'iou_threshold': 0.15,
            'use_paddleocr': False,  # Use EasyOCR
            'imgsz': 640,
            'use_local_semantics': True,
            'scale_img': False,
            'batch_size': 128
        }
        
        omniparser = Omniparser(config_easyocr)
        start_time = time.time()
        
        result = omniparser.process_single_image(
            image_path=image_path,
            output_dir=output_dir,
            datetime_prefix="troubleshoot_easyocr",
            save_outputs=True
        )
        
        processing_time = time.time() - start_time
        
        results["full_processing_easyocr"] = {
            "status": "SUCCESS",
            "processing_time": processing_time,
            "num_elements": result['num_elements'],
            "num_ocr_text": result['num_ocr_text'],
            "output_files": {
                "original_copy": result.get('original_copy_path'),
                "marked_up": result.get('marked_up_path'),
                "json": result.get('json_path')
            }
        }
        print(f"  ✓ Full processing with EasyOCR successful: {result['num_elements']} elements in {processing_time:.2f}s")
        
    except Exception as e:
        error_info = {
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        results["full_processing_easyocr"] = error_info
        print(f"  ✗ Full processing with EasyOCR failed: {type(e).__name__}: {str(e)}")
    
    return results


def generate_troubleshooting_report(image_path: Path, image_properties: Dict, ocr_results: Dict, full_processing_results: Dict) -> Dict[str, Any]:
    """Generate a comprehensive troubleshooting report"""
    
    report = {
        "image_info": {
            "filename": image_path.name,
            "file_path": str(image_path),
            "properties": image_properties
        },
        "ocr_testing": ocr_results,
        "full_processing_testing": full_processing_results,
        "analysis": {
            "potential_issues": [],
            "recommendations": [],
            "working_configurations": []
        }
    }
    
    # Analyze OCR results
    successful_ocr = []
    failed_ocr = []
    
    for test_name, result in ocr_results.items():
        if result.get("status") == "SUCCESS":
            successful_ocr.append(test_name)
        else:
            failed_ocr.append(test_name)
    
    # Analyze full processing results
    successful_processing = []
    failed_processing = []
    
    for test_name, result in full_processing_results.items():
        if result.get("status") == "SUCCESS":
            successful_processing.append(test_name)
        else:
            failed_processing.append(test_name)
    
    # Generate analysis
    if failed_ocr:
        report["analysis"]["potential_issues"].append({
            "type": "OCR_FAILURE",
            "description": f"OCR failed in {len(failed_ocr)} configurations",
            "failed_tests": failed_ocr
        })
    
    if successful_ocr:
        report["analysis"]["working_configurations"].extend(successful_ocr)
    
    if successful_processing:
        report["analysis"]["working_configurations"].extend(successful_processing)
    
    # Generate recommendations
    if "easyocr" in successful_ocr and "paddleocr_default" in failed_ocr:
        report["analysis"]["recommendations"].append({
            "priority": "HIGH",
            "action": "Use EasyOCR instead of PaddleOCR for this image",
            "reason": "EasyOCR works while PaddleOCR fails"
        })
    
    if image_properties.get("file_size_mb", 0) > 50:
        report["analysis"]["recommendations"].append({
            "priority": "MEDIUM",
            "action": "Consider resizing large images before processing",
            "reason": f"Image file size is {image_properties.get('file_size_mb', 0):.1f}MB"
        })
    
    if image_properties.get("total_pixels", 0) > 10000000:  # 10MP
        report["analysis"]["recommendations"].append({
            "priority": "MEDIUM",
            "action": "Consider resizing high-resolution images",
            "reason": f"Image has {image_properties.get('total_pixels', 0):,} pixels"
        })
    
    if successful_processing:
        report["analysis"]["recommendations"].append({
            "priority": "LOW",
            "action": "Use working configuration for batch processing",
            "reason": f"Full processing works with: {', '.join(successful_processing)}"
        })
    
    return report


def main():
    """Main troubleshooting function"""
    input_dir = Path("./imgs/tsserrors")
    output_dir = Path("./output/tsserrors")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files in the errors directory
    png_files = list(input_dir.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to troubleshoot")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    all_reports = {}
    
    for i, png_file in enumerate(png_files, 1):
        print(f"\n{'='*60}")
        print(f"TROUBLESHOOTING IMAGE {i}/{len(png_files)}: {png_file.name}")
        print(f"{'='*60}")
        
        # Analyze image properties
        print("Analyzing image properties...")
        image_properties = analyze_image_properties(png_file)
        print(f"  File size: {image_properties.get('file_size_mb', 0):.2f}MB")
        print(f"  Image size: {image_properties.get('image_size', 'Unknown')}")
        print(f"  Memory usage: {image_properties.get('memory_usage_mb', 0):.2f}MB")
        
        # Test OCR with different configurations
        print("\nTesting OCR configurations...")
        ocr_results = test_ocr_with_different_configs(png_file)
        
        # Test full Omniparser processing
        print("\nTesting full Omniparser processing...")
        full_processing_results = test_full_omniparser_processing(png_file, output_dir)
        
        # Generate report
        report = generate_troubleshooting_report(png_file, image_properties, ocr_results, full_processing_results)
        all_reports[png_file.name] = report
        
        print(f"\nTroubleshooting completed for {png_file.name}")
    
    # Save comprehensive report
    import json
    report_path = output_dir / "troubleshooting_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("TROUBLESHOOTING SUMMARY")
    print(f"{'='*80}")
    print(f"Images processed: {len(png_files)}")
    print(f"Detailed report saved to: {report_path}")
    
    # Print summary for each image
    for filename, report in all_reports.items():
        print(f"\n{filename}:")
        working_configs = report["analysis"]["working_configurations"]
        if working_configs:
            print(f"  ✓ Working configurations: {', '.join(working_configs)}")
        else:
            print(f"  ✗ No working configurations found")
        
        recommendations = report["analysis"]["recommendations"]
        if recommendations:
            print(f"  Recommendations:")
            for rec in recommendations:
                print(f"    - {rec['action']} ({rec['priority']} priority)")


if __name__ == "__main__":
    main() 