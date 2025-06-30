#!/usr/bin/env python3
"""
Example script demonstrating how to use the modularized Omniparser

This script shows how to:
1. Initialize Omniparser with custom configuration
2. Process a single image
3. Process an image from base64 string
4. Save outputs to files
"""

from pathlib import Path
from util.omniparser import Omniparser


def example_basic_usage():
    """Example of basic Omniparser usage with default configuration"""
    print("=== Basic Usage Example ===")
    
    # Initialize with default configuration
    omniparser = Omniparser()
    
    # Example image path (replace with your actual image path)
    image_path = Path("./imgs/example.png")
    
    if image_path.exists():
        # Process single image and save outputs
        result = omniparser.process_single_image(
            image_path=image_path,
            output_dir=Path("./output"),
            save_outputs=True
        )
        
        print(f"Processed image: {result['image_name']}")
        print(f"Detected {result['num_elements']} elements")
        print(f"Found {result['num_ocr_text']} OCR text items")
        print(f"Image size: {result['image_size']}")
        
        return result
    else:
        print(f"Example image not found at {image_path}")
        return None


def example_custom_configuration():
    """Example of using Omniparser with custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = {
        'som_model_path': 'weights/icon_detect/model.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': 0.25,  # Higher threshold for fewer detections
        'iou_threshold': 0.3,  # Higher IoU threshold
        'use_paddleocr': False,  # Use EasyOCR instead
        'imgsz': 800,  # Different image size
        'use_local_semantics': True,
        'scale_img': False,
        'batch_size': 64
    }
    
    # Initialize with custom configuration
    omniparser = Omniparser(config)
    
    # Example image path
    image_path = Path("./imgs/example.png")
    
    if image_path.exists():
        # Process without saving files (just get results)
        result = omniparser.process_single_image(
            image_path=image_path,
            save_outputs=False  # Don't save files, just return results
        )
        
        print(f"Processed with custom config:")
        print(f"  Box threshold: {config['BOX_TRESHOLD']}")
        print(f"  IoU threshold: {config['iou_threshold']}")
        print(f"  Detected elements: {result['num_elements']}")
        print(f"  OCR text items: {result['num_ocr_text']}")
        
        return result
    else:
        print(f"Example image not found at {image_path}")
        return None


def example_base64_processing():
    """Example of processing base64 encoded image"""
    print("\n=== Base64 Processing Example ===")
    
    omniparser = Omniparser()
    
    # Example: Load image and convert to base64
    image_path = Path("./imgs/example.png")
    
    if image_path.exists():
        from PIL import Image
        import base64
        import io
        
        # Load image and convert to base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
        
        # Process base64 image
        annotated_image_base64, parsed_content = omniparser.parse_base64(base64_string)
        
        print(f"Processed base64 image:")
        print(f"  Annotated image length: {len(annotated_image_base64)} characters")
        print(f"  Parsed content items: {len(parsed_content)}")
        
        # Optionally save the annotated image
        annotated_image_data = base64.b64decode(annotated_image_base64)
        annotated_image = Image.open(io.BytesIO(annotated_image_data))
        output_path = Path("./output/annotated_from_base64.png")
        output_path.parent.mkdir(exist_ok=True)
        annotated_image.save(output_path)
        print(f"  Saved annotated image to: {output_path}")
        
        return parsed_content
    else:
        print(f"Example image not found at {image_path}")
        return None


def example_pil_image_processing():
    """Example of processing PIL Image object directly"""
    print("\n=== PIL Image Processing Example ===")
    
    omniparser = Omniparser()
    
    # Example image path
    image_path = Path("./imgs/example.png")
    
    if image_path.exists():
        from PIL import Image
        
        # Load image as PIL Image
        pil_image = Image.open(image_path)
        
        # Process PIL Image directly
        annotated_image_base64, parsed_content = omniparser.parse_image(pil_image)
        
        print(f"Processed PIL Image:")
        print(f"  Original image size: {pil_image.size}")
        print(f"  Annotated image length: {len(annotated_image_base64)} characters")
        print(f"  Parsed content items: {len(parsed_content)}")
        
        return parsed_content
    else:
        print(f"Example image not found at {image_path}")
        return None


def main():
    """Run all examples"""
    print("Omniparser Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_configuration()
    example_base64_processing()
    example_pil_image_processing()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use Omniparser in your own code:")
    print("1. Import: from util.omniparser import Omniparser")
    print("2. Initialize: omniparser = Omniparser()")
    print("3. Process: result = omniparser.process_single_image(image_path)")


if __name__ == "__main__":
    main() 