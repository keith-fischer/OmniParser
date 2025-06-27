#!/usr/bin/env python3
"""
Post-processing script for OmniParser JSON output files.
Consolidates label_coordinates into detected_elements for easier processing.
"""

import json
import glob
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from click.decorators import pass_context


def json_consolidate(
    input_folder: str = "./output",
    file_filter: str = "*_frame_00??_elements.json", 
    output_folder: str = "./output/frames/post_processed"
) -> None:
    """
    Consolidate JSON files by merging label_coordinates into detected_elements.
    
    Args:
        input_folder: Path to input folder containing JSON files
        file_filter: Glob pattern to filter JSON files (e.g., "*elements.json")
        output_folder: Path to output folder for processed files
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files matching the filter recursively
    search_pattern = os.path.join(input_folder, "**", file_filter)
    json_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(json_files)} JSON files matching pattern: {search_pattern}")
    
    for json_file in json_files:
        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process the file
            processed_data = process_json_data(data)
            
            # Generate output filename
            input_filename = os.path.basename(json_file)
            output_filename = input_filename.replace("elements", "post")
            output_path = os.path.join(output_folder, output_filename)
            
            # Write processed data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            print(f"Processed: {input_filename} -> {output_filename}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


def process_json_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single JSON data object by merging label_coordinates and ocr_bbox into detected_elements.
    
    Args:
        data: The JSON data dictionary
        
    Returns:
        Processed data with label_coordinates and ocr_bbox merged into detected_elements
    """
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Check if detected_elements exists
    if 'detected_elements' not in processed_data:
        print("Warning: Missing 'detected_elements' in data")
        return processed_data
    
    detected_elements = processed_data['detected_elements']
    
    # Process label_coordinates if it exists
    if 'label_coordinates' in processed_data:
        label_coords = processed_data['label_coordinates']
        print(f"Processing {len(label_coords)} label coordinates for {len(detected_elements)} detected elements")
        
        # Iterate through label_coordinates
        for str_index, coord_data in label_coords.items():
            try:
                # Convert string index to integer
                int_index = int(str_index)
                
                # Check if index is valid
                if int_index < len(detected_elements):
                    # Copy the coordinate data to the detected element
                    detected_elements[int_index]['label_coordinates'] = coord_data.copy()
                    print(f"  Added label_coordinates to detected_elements[{int_index}]")
                else:
                    print(f"  Warning: Index {int_index} out of range for detected_elements (max: {len(detected_elements)-1})")
                    
            except ValueError as e:
                print(f"  Error: Could not convert '{str_index}' to integer: {e}")
            except Exception as e:
                print(f"  Error processing index {str_index}: {e}")
    
    # Process ocr_bbox if it exists
    if 'ocr_bbox' in processed_data:
        ocr_bbox_list = processed_data['ocr_bbox']
        print(f"Processing {len(ocr_bbox_list)} ocr_bbox items for {len(detected_elements)} detected elements")
        
        # Iterate through ocr_bbox list (same index as detected_elements)
        for index, bbox_data in enumerate(ocr_bbox_list):
            try:
                # Check if index is valid
                if index < len(detected_elements):
                    # Copy the bbox data to the detected element
                    detected_elements[index]['ocr_bbox'] = bbox_data.copy()
                    print(f"  Added ocr_bbox to detected_elements[{index}]")
                else:
                    print(f"  Warning: Index {index} out of range for detected_elements (max: {len(detected_elements)-1})")
                    
            except Exception as e:
                print(f"  Error processing ocr_bbox index {index}: {e}")
    
    return processed_data


def run_json_consolidate():
    """Main function to run the consolidation process."""
    # Example usage
    input_folder = "./output"
    file_filter = "*_frame_00??_elements.json"
    output_folder = "./output/frames/post_processed"
    
    print("Starting JSON consolidation process...")
    print(f"Input folder: {input_folder}")
    print(f"File filter: {file_filter}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    json_consolidate(input_folder, file_filter, output_folder)
    
    print("-" * 50)
    print("Consolidation process completed!")

def render_json_rectangles(json_path: str, png_path: str):
    """Render JSON rectangles to images."""
    try:
        # Load the PNG image
        print(f"Loading image: {png_path}")
        image = cv2.imread(png_path)
        if image is None:
            print(f"Error: Could not load image from {png_path}")
            return
        
        print(f"Image loaded successfully. Shape: {image.shape}")
        img_height, img_width = image.shape[:2]
        
        # Load the JSON data
        print(f"Loading JSON: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if detected_elements exists
        if 'detected_elements' not in data:
            print("Error: No 'detected_elements' found in JSON")
            return
        
        detected_elements = data['detected_elements']
        print(f"Found {len(detected_elements)} detected elements")
        
        # Iterate through detected elements and draw rectangles
        for i, element in enumerate(detected_elements):
            try:
                coords = None
                # Prefer ocr_bbox (absolute pixel)
                if 'ocr_bbox' in element and isinstance(element['ocr_bbox'], list) and len(element['ocr_bbox']) == 4:
                    coords = element['ocr_bbox']
                    x1, y1, x2, y2 = map(int, coords)
                # Fallback to bbox (normalized)
                elif 'bbox' in element and isinstance(element['bbox'], list) and len(element['bbox']) == 4:
                    bx = element['bbox']
                    x1 = int(bx[0] * img_width)
                    y1 = int(bx[1] * img_height)
                    x2 = int(bx[2] * img_width)
                    y2 = int(bx[3] * img_height)
                # Fallback to label_coordinates (normalized)
                elif 'label_coordinates' in element and isinstance(element['label_coordinates'], list) and len(element['label_coordinates']) == 4:
                    bx = element['label_coordinates']
                    x1 = int(bx[0] * img_width)
                    y1 = int(bx[1] * img_height)
                    x2 = int((bx[0] + bx[2]) * img_width)
                    y2 = int((bx[1] + bx[3]) * img_height)
                else:
                    print(f"  Skipping element {i}: No valid bbox/ocr_bbox/label_coordinates found")
                    continue
                # Draw red rectangle (BGR format: red = (0, 0, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                print(f"  Drew rectangle for element {i}: ({x1}, {y1}) to ({x2}, {y2})")
            except Exception as e:
                print(f"  Error processing element {i}: {e}")
                continue
        
        # Display the image
        print("Displaying image with rectangles. Press any key to close.")
        cv2.imshow('Detected Elements', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save the annotated image
        output_path = png_path.replace('.png', '_annotated.png')
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error in render_json_rectangles: {e}")

def render_json():
    # Example usage
    json_path = "/Users/fischtech/repos/github/OmniParser/output/post_processed/frames/post_processed/20250625_183836_frame_0001_post.json"
    png_path = "/Users/fischtech/repos/github/OmniParser/output/frame_0001/20250625_183836_frame_0001_original.png"
    render_json_rectangles(json_path, png_path)
    
def main(mode: int = 0):
    if mode == 0:   
        run_json_consolidate()
    elif mode == 1:
        render_json()

if __name__ == "__main__":
    main(1) 