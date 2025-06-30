#!/usr/bin/env python3
"""
Test script for Omniparser module

This script demonstrates how to test the Omniparser module with default parameters
and validate that all functionality is working correctly.
"""

from pathlib import Path
from util.omniparser import Omniparser, test_omniparser_module


def test_with_default_configuration():
    """Test Omniparser with default configuration"""
    print("Testing Omniparser with default configuration...")
    
    try:
        # Initialize with default configuration
        omniparser = Omniparser()
        
        # Run the test
        test_results = omniparser.test_module()
        
        print(f"\nTest Status: {test_results['test_status']}")
        
        if test_results['test_status'] == 'SUCCESS':
            print("✓ All tests passed!")
            print(f"Test image: {test_results['test_image_path']}")
            print(f"Image size: {test_results['image_size']}")
            print(f"Elements detected: {test_results['results']['num_elements']}")
            print(f"OCR text items: {test_results['results']['num_ocr_text']}")
            
            if test_results.get('output_files_saved'):
                print(f"Output saved to: {test_results['output_directory']}")
        
        return test_results
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        return {"test_status": "FAILED", "error": str(e)}


def test_with_custom_configuration():
    """Test Omniparser with custom configuration"""
    print("\nTesting Omniparser with custom configuration...")
    
    # Custom configuration for testing
    custom_config = {
        'som_model_path': 'weights/icon_detect/model.pt',
        'caption_model_name': 'florence2',
        'caption_model_path': 'weights/icon_caption_florence',
        'BOX_TRESHOLD': 0.25,  # Higher threshold
        'iou_threshold': 0.3,  # Higher IoU threshold
        'use_paddleocr': True,  # Use EasyOCR
        'imgsz': 3000,  # Different image size
        'use_local_semantics': True,
        'scale_img': True,
        'batch_size': 64
    }
    
    try:
        # Initialize with custom configuration
        omniparser = Omniparser(custom_config)
        
        # Run the test
        test_results = omniparser.test_module(save_test_output=False)  # Don't save files
        
        print(f"\nTest Status: {test_results['test_status']}")
        
        if test_results['test_status'] == 'SUCCESS':
            print("✓ Custom configuration test passed!")
            print(f"Configuration used: {custom_config}")
            print(f"Elements detected: {test_results['results']['num_elements']}")
            print(f"OCR text items: {test_results['results']['num_ocr_text']}")
        
        return test_results
        
    except Exception as e:
        print(f"✗ Custom configuration test failed: {str(e)}")
        return {"test_status": "FAILED", "error": str(e)}


def test_with_specific_image(image_path: str):
    """Test Omniparser with a specific image"""
    print(f"\nTesting Omniparser with specific image: {image_path}")
    
    try:
        # Use the standalone test function
        test_results = test_omniparser_module(test_image_path=image_path)
        
        print(f"\nTest Status: {test_results['test_status']}")
        
        if test_results['test_status'] == 'SUCCESS':
            print("✓ Specific image test passed!")
            print(f"Test image: {test_results['test_image_path']}")
            print(f"Elements detected: {test_results['results']['num_elements']}")
            print(f"OCR text items: {test_results['results']['num_ocr_text']}")
        
        return test_results
        
    except Exception as e:
        print(f"✗ Specific image test failed: {str(e)}")
        return {"test_status": "FAILED", "error": str(e)}


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("OMNIPARSER COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    all_results = []
    
    # Test 1: Default configuration
    print("\n1. Testing default configuration...")
    result1 = test_with_default_configuration()
    all_results.append(("Default Configuration", result1))
    
    # Test 2: Custom configuration
    print("\n2. Testing custom configuration...")
    result2 = test_with_custom_configuration()
    all_results.append(("Custom Configuration", result2))
    
    # Test 3: Specific image (if available)
    test_image_path = "./imgs/example.png"
    if Path(test_image_path).exists():
        print(f"\n3. Testing with specific image: {test_image_path}")
        result3 = test_with_specific_image(test_image_path)
        all_results.append(("Specific Image", result3))
    else:
        print(f"\n3. Skipping specific image test (not found: {test_image_path})")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = len(all_results)
    
    for test_name, result in all_results:
        status = result.get('test_status', 'UNKNOWN')
        if status == 'SUCCESS':
            successful_tests += 1
            print(f"✓ {test_name}: PASSED")
        else:
            print(f"✗ {test_name}: FAILED")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    
    print(f"\nOverall Result: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED - OMNIPARSER MODULE IS WORKING CORRECTLY!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_results


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        # Test with specific image if provided
        image_path = sys.argv[1]
        test_with_specific_image(image_path)
    else:
        # Run comprehensive test suite
        run_comprehensive_test()


if __name__ == "__main__":
    main() 