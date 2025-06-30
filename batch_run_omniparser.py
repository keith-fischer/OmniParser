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
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Import the modularized Omniparser
from util.omniparser import Omniparser


def get_datetime_prefix():
    """Get current datetime as YYYYMMDD_HHMMSS string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_omniparser_config(
    box_threshold: float = 0.15,
    iou_threshold: float = 0.15,
    use_paddleocr: bool = False,
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


def process_batch(
    input_dir: Path,
    img_filter: str,
    output_dir: Path,
    datetime_prefix: str,
    box_threshold: float = 0.10,
    iou_threshold: float = 0.10,
    use_paddleocr: bool = False,
    imgsz: int = 1000
) -> List[Dict[str, Any]]:
    """
    Process all PNG images in the input directory
    
    Args:
        input_dir: Directory containing input images
        img_filter: File filter pattern (e.g., "*.png")
        output_dir: Directory to save outputs
        datetime_prefix: Prefix for output filenames
        box_threshold: Confidence threshold for bounding box detection
        iou_threshold: IoU threshold for non-maximum suppression
        use_paddleocr: Whether to use PaddleOCR instead of EasyOCR
        imgsz: Input image size for YOLO model
        
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
    
    # Create configuration with all the parameters
    config = create_omniparser_config(
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        imgsz=imgsz
    )
    
    # Initialize Omniparser with the configuration
    omniparser = Omniparser(config)
    
    # Process each image
    all_results = []
    for i, png_file in enumerate(png_files, 1):
        print(f"\n[{i}/{len(png_files)}] Processing: {png_file.name}")
        
        try:
            result = omniparser.process_single_image(
                image_path=png_file,
                output_dir=output_dir,
                datetime_prefix=datetime_prefix,
                save_outputs=True
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {png_file.name}: {str(e)}")
            # Add error result to maintain indexing
            all_results.append({
                "image_name": png_file.stem,
                "error": str(e),
                "num_elements": 0,
                "num_ocr_text": 0
            })
    
    return all_results


def save_batch_summary(results: List[Dict[str, Any]], output_dir: Path, datetime_prefix: str):
    """Save a summary of batch processing results"""
    summary_path = output_dir / f"{datetime_prefix}_batch_summary.json"
    
    summary_data = {
        "batch_info": {
            "total_images": len(results),
            "successful_processing": len([r for r in results if "error" not in r]),
            "failed_processing": len([r for r in results if "error" in r]),
            "datetime_prefix": datetime_prefix
        },
        "results": results
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Batch summary saved: {summary_path}")


def create_ascii_table_report(all_results: List[Dict[str, Any]]) -> str:
    """Create an ASCII table report of processing results"""
    if not all_results:
        return "No results to report"
    
    # Filter out error results for statistics
    successful_results = [r for r in all_results if "error" not in r]
    
    if not successful_results:
        return "No successful processing results to report"
    
    # Calculate statistics
    total_elements = sum(r.get("num_elements", 0) for r in successful_results)
    total_ocr_text = sum(r.get("num_ocr_text", 0) for r in successful_results)
    avg_elements = total_elements / len(successful_results) if successful_results else 0
    avg_ocr_text = total_ocr_text / len(successful_results) if successful_results else 0
    
    # Create table
    table = []
    table.append("=" * 80)
    table.append("BATCH PROCESSING SUMMARY")
    table.append("=" * 80)
    table.append(f"Total Images Processed: {len(all_results)}")
    table.append(f"Successful: {len(successful_results)}")
    table.append(f"Failed: {len(all_results) - len(successful_results)}")
    table.append("")
    
    if successful_results:
        table.append("SUCCESSFUL PROCESSING STATISTICS:")
        table.append("-" * 40)
        table.append(f"Total Elements Detected: {total_elements}")
        table.append(f"Total OCR Text Items: {total_ocr_text}")
        table.append(f"Average Elements per Image: {avg_elements:.2f}")
        table.append(f"Average OCR Text per Image: {avg_ocr_text:.2f}")
        table.append("")
    
    # Individual image results
    table.append("INDIVIDUAL IMAGE RESULTS:")
    table.append("-" * 80)
    table.append(f"{'Image Name':<30} {'Elements':<10} {'OCR Text':<10} {'Status':<10}")
    table.append("-" * 80)
    
    for result in all_results:
        image_name = result.get("image_name", "Unknown")
        if "error" in result:
            table.append(f"{image_name:<30} {'N/A':<10} {'N/A':<10} {'ERROR':<10}")
        else:
            elements = result.get("num_elements", 0)
            ocr_text = result.get("num_ocr_text", 0)
            table.append(f"{image_name:<30} {elements:<10} {ocr_text:<10} {'SUCCESS':<10}")
    
    table.append("=" * 80)
    
    return "\n".join(table)


def print_summary_statistics(all_results: List[Dict[str, Any]]):
    """Print summary statistics to console"""
    if not all_results:
        print("No results to analyze")
        return
    
    # Filter out error results for statistics
    successful_results = [r for r in all_results if "error" not in r]
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total Images: {len(all_results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(all_results) - len(successful_results)}")
    
    if successful_results:
        total_elements = sum(r.get("num_elements", 0) for r in successful_results)
        total_ocr_text = sum(r.get("num_ocr_text", 0) for r in successful_results)
        
        print(f"\nTotal Elements Detected: {total_elements}")
        print(f"Total OCR Text Items: {total_ocr_text}")
        print(f"Average Elements per Image: {total_elements / len(successful_results):.2f}")
        print(f"Average OCR Text per Image: {total_ocr_text / len(successful_results):.2f}")
    
    # Show failed images
    failed_results = [r for r in all_results if "error" in r]
    if failed_results:
        print(f"\nFailed Images:")
        for result in failed_results:
            print(f"  - {result.get('image_name', 'Unknown')}: {result.get('error', 'Unknown error')}")


def print_accumulated_stats(all_results: List[Dict[str, Any]], current_box: float, current_iou: float):
    """Print accumulated statistics for current threshold settings"""
    if not all_results:
        return
    
    successful_results = [r for r in all_results if "error" not in r]
    
    if successful_results:
        total_elements = sum(r.get("num_elements", 0) for r in successful_results)
        total_ocr_text = sum(r.get("num_ocr_text", 0) for r in successful_results)
        
        print(f"\n[Box: {current_box:.2f}, IoU: {current_iou:.2f}] "
              f"Images: {len(successful_results)}/{len(all_results)}, "
              f"Elements: {total_elements}, OCR: {total_ocr_text}")


def run_omniparse(
    img_path: str = "./imgs",
    img_filter: str = "*.png",
    out_dir: str = "./output",
    box: float = 0.05, 
    iou: float = 0.05,
    img_size: int = 3000
):
    """
    Main function to run OmniParser batch processing
    
    Args:
        img_path: Path to input images directory
        img_filter: File filter pattern
        out_dir: Output directory path
        box: Box threshold for detection
        iou: IoU threshold for detection
        img_size: Input image size
    """
    input_dir = Path(img_path)
    output_dir = Path(out_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get datetime prefix for this batch
    datetime_prefix = get_datetime_prefix()
    
    print(f"Starting OmniParser batch processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"File filter: {img_filter}")
    print(f"Box threshold: {box}")
    print(f"IoU threshold: {iou}")
    print(f"Image size: {img_size}")
    
    # Process the batch
    results = process_batch(
        input_dir=input_dir,
        img_filter=img_filter,
        output_dir=output_dir,
        datetime_prefix=datetime_prefix,
        box_threshold=box,
        iou_threshold=iou,
        use_paddleocr=False,
        imgsz=img_size
    )
    
    # Save batch summary
    save_batch_summary(results, output_dir, datetime_prefix)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Print ASCII table report
    report = create_ascii_table_report(results)
    print(f"\n{report}")
    
    return results


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(
        description="Batch process PNG images with OmniParser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_run_omniparser.py --input_dir ./images --output_dir ./output
  python batch_run_omniparser.py --input_dir ./screenshots --box 0.1 --iou 0.1
  python batch_run_omniparser.py --input_dir ./test --img_size 2000
        """
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        default=".imgs/tsserrors",
        help="Input directory containing PNG images (default: ./imgs)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./output/tsserrors",
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
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    # Run the batch processing
    try:
        results = run_omniparse(
            img_path=args.input_dir,
            img_filter=args.img_filter,
            out_dir=args.output_dir,
            box=args.box,
            iou=args.iou,
            img_size=args.img_size
        )
        
        print(f"\nBatch processing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 