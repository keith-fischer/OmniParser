#!/usr/bin/env python3
"""
Batch OmniParser Processing Script

This script processes multiple PNG images in a directory using OmniParser and generates:
1. Copy of original PNG file
2. Marked up PNG with detected icon regions
3. JSON file with all detected elements

Usage:
    python batch_run_omniparser.py --input_dir /path/to/input/images --output_dir /path/to/output
"""

import argparse
import os
import json
import base64
import io
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import torch
from PIL import Image
import numpy as np

# Import OmniParser utilities
from util.utils import (
    check_ocr_box, 
    get_yolo_model, 
    get_caption_model_processor, 
    get_som_labeled_img
)


def get_datetime_prefix():
    """Get current datetime as YYYYMMDD_HHMMSS string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_models():
    """Initialize the YOLO and caption models"""
    print("Loading YOLO model...")
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    
    print("Loading caption model...")
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence"
    )
    
    return yolo_model, caption_model_processor


def process_single_image(
    image_path: Path,
    output_dir: Path,
    datetime_prefix: str,
    yolo_model,
    caption_model_processor,
    box_threshold: float = 0.15,
    iou_threshold: float = 0.15,
    use_paddleocr: bool = True,
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Process a single image and save the three output files
    
    Returns:
        Dict containing processing results and file paths
    """
    print(f"Processing: {image_path.name}")
    
    # Create output subdirectory for this image
    image_name = image_path.stem
    image_output_dir = output_dir / image_name
    image_output_dir.mkdir(exist_ok=True)
    
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
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    # Process with OmniParser
    dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold, 
        imgsz=imgsz
    )
    
    # Convert thresholds to integers for filename (multiply by 100 and remove decimal)
    box_threshold_int = int(box_threshold * 100)
    iou_threshold_int = int(iou_threshold * 100)
    
    # Save the three output files with datetime prefix
    
    # 1. Copy of original PNG
    original_copy_path = image_output_dir / f"{datetime_prefix}_{image_name}_original.png"
    image_input.save(original_copy_path)
    
    # 2. Marked up PNG with detected regions (includes "omni" and threshold values in filename)
    marked_up_path = image_output_dir / f"{datetime_prefix}_{image_name}_omni_b{box_threshold_int}_i{iou_threshold_int}.png"
    # Convert base64 encoded image back to PIL Image and save
    marked_up_image = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
    marked_up_image.save(marked_up_path)
    
    # 3. JSON file with all detected elements
    json_path = image_output_dir / f"{datetime_prefix}_{image_name}_elements.json"
    
    # Prepare JSON data
    json_data = {
        "image_info": {
            "original_path": str(image_path),
            "image_size": image_input.size,
            "processing_parameters": {
                "box_threshold": box_threshold,
                "iou_threshold": iou_threshold,
                "use_paddleocr": use_paddleocr,
                "imgsz": imgsz
            }
        },
        "detected_elements": parsed_content_list,
        "label_coordinates": label_coordinates,
        "ocr_text": text,
        "ocr_bbox": ocr_bbox
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved: {original_copy_path}")
    print(f"  ✓ Saved: {marked_up_path}")
    print(f"  ✓ Saved: {json_path}")
    
    return {
        "image_name": image_name,
        "original_copy_path": str(original_copy_path),
        "marked_up_path": str(marked_up_path),
        "json_path": str(json_path),
        "num_elements": len(parsed_content_list),
        "num_ocr_text": len(text)
    }


def process_batch(
    input_dir: Path,
    output_dir: Path,
    datetime_prefix: str,
    box_threshold: float = 0.10,
    iou_threshold: float = 0.10,
    use_paddleocr: bool = True,
    imgsz: int = 1000
) -> List[Dict[str, Any]]:
    """
    Process all PNG images in the input directory
    
    Returns:
        List of processing results for each image
    """
    # Find all PNG files
    png_files = list(input_dir.glob("*TSS-TC-16-screen8.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return []
    
    print(f"Found {len(png_files)} PNG files to process")
    print(f"Using datetime prefix: {datetime_prefix}")
    
    # Setup models
    yolo_model, caption_model_processor = setup_models()
    
    # Process each image
    results = []
    for i, png_file in enumerate(png_files, 1):
        print(f"\n[{i}/{len(png_files)}] Processing {png_file.name}")
        
        try:
            result = process_single_image(
                png_file,
                output_dir,
                datetime_prefix,
                yolo_model,
                caption_model_processor,
                box_threshold,
                iou_threshold,
                use_paddleocr,
                imgsz
            )
            results.append(result)
            print(f"  ✓ Successfully processed {png_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {png_file.name}: {str(e)}")
            results.append({
                "image_name": png_file.stem,
                "error": str(e),
                "status": "failed"
            })
    
    return results


def save_batch_summary(results: List[Dict[str, Any]], output_dir: Path, datetime_prefix: str):
    """Save a summary of all processing results"""
    summary_path = output_dir / f"{datetime_prefix}_batch_processing_summary.json"
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    summary = {
        "processing_summary": {
            "total_images": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "datetime_prefix": datetime_prefix,
            "processing_timestamp": datetime.now().isoformat()
        },
        "successful_results": successful,
        "failed_results": failed
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch processing summary saved to: {summary_path}")
    print(f"Total images: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")


def run_omniparse(box: float, iou: float):
    parser = argparse.ArgumentParser(description="Batch process PNG images with OmniParser")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/Users/fischtech/repos/github/ollamapoc/image/TSS",
        help="Directory containing PNG images to process (default: /Users/fischtech/repos/github/OmniParser/imgs)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/Users/fischtech/repos/github/OmniParser/output",
        help="Directory to save processed results (default: /Users/fischtech/repos/github/OmniParser/output)"
    )
    parser.add_argument(
        "--box_threshold", 
        type=float, 
        default=box,
        help="Box detection threshold (default: 0.15)"
    )
    parser.add_argument(
        "--iou_threshold", 
        type=float, 
        default=iou,
        help="IoU threshold for overlap removal (default: 0.15)"
    )
    parser.add_argument(
        "--use_paddleocr", 
        action="store_true",
        default=True,
        help="Use PaddleOCR for text detection (default: True)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=3000,
        help="Image size for processing (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Override with the passed parameters
    args.box_threshold = box
    args.iou_threshold = iou
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Get datetime prefix for this batch run
    datetime_prefix = get_datetime_prefix()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Datetime prefix: {datetime_prefix}")
    print(f"Processing parameters:")
    print(f"  Box threshold: {args.box_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Use PaddleOCR: {args.use_paddleocr}")
    print(f"  Image size: {args.imgsz}")
    
    # Process the batch
    results = process_batch(
        input_dir,
        output_dir,
        datetime_prefix,
        args.box_threshold,
        args.iou_threshold,
        args.use_paddleocr,
        args.imgsz
    )
    
    # Save summary
    save_batch_summary(results, output_dir, datetime_prefix)
    
    print("\nBatch processing completed!")
    return results

def create_ascii_table_report(all_results: List[Dict[str, Any]]) -> str:
    """
    Create an ASCII table report from the processing results
    """
    # Filter successful results that have the required data
    valid_results = []
    for result in all_results:
        if "error" not in result and "test_parameters" in result and "num_elements" in result and "num_ocr_text" in result:
            valid_results.append(result)
    
    if not valid_results:
        return "No valid results to display in table format."
    
    # Create table headers
    headers = ["Box", "IoU", "Detected Elements", "OCR Text", "Image Name"]
    
    # Prepare table data
    table_data = []
    for result in valid_results:
        box_threshold = result["test_parameters"]["box_threshold"]
        iou_threshold = result["test_parameters"]["iou_threshold"]
        num_elements = result["num_elements"]
        num_ocr_text = result["num_ocr_text"]
        image_name = result["image_name"]
        
        table_data.append([
            f"{box_threshold:.2f}",
            f"{iou_threshold:.2f}",
            str(num_elements),
            str(num_ocr_text),
            image_name
        ])
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in table_data:
            max_width = max(max_width, len(row[i]))
        col_widths.append(max_width + 2)  # Add padding
    
    # Build the table
    table_lines = []
    
    # Header row
    header_line = "|"
    separator_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {header:<{col_widths[i]-1}}|"
        separator_line += f" {'-' * (col_widths[i]-1)} |"
    
    table_lines.append(header_line)
    table_lines.append(separator_line)
    
    # Data rows
    for row in table_data:
        data_line = "|"
        for i, cell in enumerate(row):
            data_line += f" {cell:<{col_widths[i]-1}}|"
        table_lines.append(data_line)
    
    return "\n".join(table_lines)

def print_summary_statistics(all_results: List[Dict[str, Any]]):
    """
    Print summary statistics about the processing results
    """
    # Filter successful results
    valid_results = [r for r in all_results if "error" not in r and "num_elements" in r and "num_ocr_text" in r]
    
    if not valid_results:
        print("No valid results for statistics.")
        return
    
    # Extract counts
    element_counts = [r["num_elements"] for r in valid_results]
    ocr_counts = [r["num_ocr_text"] for r in valid_results]
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    print(f"Total successful tests: {len(valid_results)}")
    print(f"Total images processed: {len(set(r['image_name'] for r in valid_results))}")
    
    print(f"\nDetected Elements Statistics:")
    print(f"  Min: {min(element_counts)}")
    print(f"  Max: {max(element_counts)}")
    print(f"  Average: {sum(element_counts) / len(element_counts):.2f}")
    print(f"  Total across all tests: {sum(element_counts)}")
    
    print(f"\nOCR Text Statistics:")
    print(f"  Min: {min(ocr_counts)}")
    print(f"  Max: {max(ocr_counts)}")
    print(f"  Average: {sum(ocr_counts) / len(ocr_counts):.2f}")
    print(f"  Total across all tests: {sum(ocr_counts)}")
    
    # Find best performing combinations
    print(f"\nTop 5 combinations by detected elements:")
    top_elements = sorted(valid_results, key=lambda x: x["num_elements"], reverse=True)[:5]
    for i, result in enumerate(top_elements, 1):
        print(f"  {i}. Box={result['test_parameters']['box_threshold']:.2f}, "
              f"IoU={result['test_parameters']['iou_threshold']:.2f}, "
              f"Elements={result['num_elements']}, "
              f"OCR={result['num_ocr_text']}")
    
    print(f"\nTop 5 combinations by OCR text:")
    top_ocr = sorted(valid_results, key=lambda x: x["num_ocr_text"], reverse=True)[:5]
    for i, result in enumerate(top_ocr, 1):
        print(f"  {i}. Box={result['test_parameters']['box_threshold']:.2f}, "
              f"IoU={result['test_parameters']['iou_threshold']:.2f}, "
              f"Elements={result['num_elements']}, "
              f"OCR={result['num_ocr_text']}")

def print_accumulated_stats(all_results: List[Dict[str, Any]], current_box: float, current_iou: float):
    """
    Print accumulated statistics and ASCII table after each combination is processed
    """
    # Filter successful results
    valid_results = [r for r in all_results if "error" not in r and "num_elements" in r and "num_ocr_text" in r]
    
    if not valid_results:
        return
    
    # Extract counts
    element_counts = [r["num_elements"] for r in valid_results]
    ocr_counts = [r["num_ocr_text"] for r in valid_results]
    
    print(f"\n📊 ACCUMULATED STATS (after Box={current_box:.2f}, IoU={current_iou:.2f}):")
    print(f"   Successful tests: {len(valid_results)}")
    print(f"   Elements - Min: {min(element_counts)}, Max: {max(element_counts)}, Avg: {sum(element_counts)/len(element_counts):.1f}")
    print(f"   OCR Text - Min: {min(ocr_counts)}, Max: {max(ocr_counts)}, Avg: {sum(ocr_counts)/len(ocr_counts):.1f}")
    
    # Show current best performers
    if len(valid_results) >= 3:
        top_elements = sorted(valid_results, key=lambda x: x["num_elements"], reverse=True)[:3]
        print(f"   Top 3 by elements: ", end="")
        for i, result in enumerate(top_elements):
            print(f"Box={result['test_parameters']['box_threshold']:.2f}/IoU={result['test_parameters']['iou_threshold']:.2f}({result['num_elements']})", end="")
            if i < 2:
                print(", ", end="")
        print()
    
    # Print the accumulated ASCII table
    print(f"\n📋 ACCUMULATED RESULTS TABLE:")
    print(f"{'='*80}")
    table_report = create_ascii_table_report(all_results)
    print(table_report)
    print(f"{'='*80}")

def main():
    """
    Main function that loops through different box and IOU threshold values
    to test their effects on processing results.
    """
    print("Starting OmniParser batch processing with parameter sweep...")
    print("Testing box_threshold and iou_threshold values from 0.01 to 0.50")
    
    # Define the range for testing
    min_threshold = 0.01
    max_threshold = 0.50
    step = 0.05  # You can adjust this step size
    
    # Create a list of threshold values to test
    threshold_values = [round(min_threshold + i * step, 2) for i in range(int((max_threshold - min_threshold) / step) + 1)]
    
    print(f"Testing {len(threshold_values)} threshold values: {threshold_values}")
    print(f"Total combinations to test: {len(threshold_values) * len(threshold_values)}")
    
    all_results = []
    combination_count = 0
    total_combinations = len(threshold_values) * len(threshold_values)
    
    # Loop through all combinations of box and IOU thresholds
    for box_threshold in threshold_values:
        for iou_threshold in threshold_values:
            combination_count += 1
            print(f"\n{'='*60}")
            print(f"Testing [{combination_count}/{total_combinations}]: box_threshold={box_threshold:.2f}, iou_threshold={iou_threshold:.2f}")
            print(f"{'='*60}")
            
            try:
                results = run_omniparse(box=box_threshold, iou=iou_threshold)
                
                # Add threshold information to results
                if results is not None:
                    for result in results:
                        if "error" not in result:
                            result["test_parameters"] = {
                                "box_threshold": box_threshold,
                                "iou_threshold": iou_threshold
                            }
                    
                    all_results.extend(results)
                else:
                    # Handle case where run_omniparse returns None
                    all_results.append({
                        "test_parameters": {
                            "box_threshold": box_threshold,
                            "iou_threshold": iou_threshold
                        },
                        "error": "run_omniparse returned None",
                        "status": "failed"
                    })
                
                print(f"✓ Completed testing with box={box_threshold:.2f}, iou={iou_threshold:.2f}")
                
                # Print accumulated statistics after each combination
                print_accumulated_stats(all_results, box_threshold, iou_threshold)
                
            except Exception as e:
                print(f"✗ Error during testing with box={box_threshold:.2f}, iou={iou_threshold:.2f}: {str(e)}")
                all_results.append({
                    "test_parameters": {
                        "box_threshold": box_threshold,
                        "iou_threshold": iou_threshold
                    },
                    "error": str(e),
                    "status": "failed"
                })
    
    # Save comprehensive summary of all tests
    print(f"\n{'='*60}")
    print("FINAL COMPREHENSIVE TESTING SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in all_results if "error" not in r]
    failed_tests = [r for r in all_results if "error" in r]
    
    print(f"Total test combinations: {total_combinations}")
    
    # Count successful combinations safely
    successful_combinations = set()
    for r in successful_tests:
        if "test_parameters" in r:
            successful_combinations.add((r['test_parameters']['box_threshold'], r['test_parameters']['iou_threshold']))
    
    # Count failed combinations safely
    failed_combinations = set()
    for r in failed_tests:
        if "test_parameters" in r:
            failed_combinations.add((r['test_parameters']['box_threshold'], r['test_parameters']['iou_threshold']))
    
    print(f"Successful combinations: {len(successful_combinations)}")
    print(f"Failed combinations: {len(failed_combinations)}")
    
    # Print final summary statistics
    print_summary_statistics(all_results)
    
    # Generate and display ASCII table report
    print(f"\n{'='*80}")
    print("DETAILED RESULTS TABLE")
    print(f"{'='*80}")
    table_report = create_ascii_table_report(all_results)
    print(table_report)
    
    # Save comprehensive results
    comprehensive_summary = {
        "testing_summary": {
            "threshold_range": {"min": min_threshold, "max": max_threshold, "step": step},
            "threshold_values_tested": threshold_values,
            "total_combinations": total_combinations,
            "successful_combinations": len(successful_combinations),
            "failed_combinations": len(failed_combinations),
            "testing_timestamp": datetime.now().isoformat()
        },
        "all_results": all_results,
        "ascii_table_report": table_report
    }
    
    output_dir = Path("/Users/fischtech/repos/github/OmniParser/output")
    comprehensive_summary_path = output_dir / f"comprehensive_testing_summary_{get_datetime_prefix()}.json"
    
    with open(comprehensive_summary_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nComprehensive testing summary saved to: {comprehensive_summary_path}")
    print("Parameter sweep testing completed!")


if __name__ == "__main__":
    main() 