#!/usr/bin/env python3
"""
Test script for the three-tier OCR fallback system

This script tests the new OCR engine with automatic fallback:
1. EasyOCR (Primary)
2. PaddleOCR (Secondary) 
3. Tesseract (Tertiary)
"""

import sys
from pathlib import Path
import time
import json

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from util.ocr_engine import OCREngine, get_global_ocr_engine


def test_ocr_engine_initialization():
    """Test OCR engine initialization"""
    print("=" * 60)
    print("TESTING OCR ENGINE INITIALIZATION")
    print("=" * 60)
    
    # Test with verbose output
    ocr_engine = OCREngine(verbose=True)
    
    print(f"Available engines: {ocr_engine.get_available_engines()}")
    print(f"Engine order: {ocr_engine.engine_order}")
    print(f"Primary engine: {ocr_engine.primary_engine}")
    
    return ocr_engine


def save_ocr_results(engine_name, texts, bboxes, out_dir="."):
    """Save OCR results to a JSON file"""
    data = {
        "engine": engine_name,
        "texts": texts,
        "bboxes": bboxes
    }
    out_path = Path(out_dir) / f"ocr_{engine_name}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved OCR results for {engine_name} to {out_path}")


def count_words(texts):
    """Count total words in a list of strings"""
    return sum(len(t.split()) for t in texts)


def compare_ocr_results(results_dict):
    """Compare word and element counts for all engines and print a table"""
    engines = list(results_dict.keys())
    word_counts = [count_words(results_dict[eng]['texts']) for eng in engines]
    elem_counts = [len(results_dict[eng]['texts']) for eng in engines]
    
    # Print table
    print("\nOCR ENGINE COMPARISON TABLE:")
    print("+--------------+" + "+".join([f"{eng:^15}" for eng in engines]) + "+")
    print(f"| {'Metric':^12} |" + "|".join([f"{eng:^15}" for eng in engines]) + "|")
    print("+--------------+" + "+".join(["---------------"]*len(engines)) + "+")
    print(f"| {'Word Count':^12} |" + "|".join([f"{wc:^15}" for wc in word_counts]) + "|")
    print(f"| {'Elem Count':^12} |" + "|".join([f"{ec:^15}" for ec in elem_counts]) + "|")
    print("+--------------+" + "+".join(["---------------"]*len(engines)) + "+")


def test_ocr_engines_individually(ocr_engine: OCREngine, test_image_path: str):
    """Test each OCR engine individually, save results, and return results dict"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL OCR ENGINES")
    print("=" * 60)
    
    results = ocr_engine.test_engines(test_image_path)
    ocr_results = {}
    
    for engine_name, result in results.items():
        print(f"\n{engine_name.upper()}:")
        if result.get("status") == "success":
            print(f"  ✓ Success: {result['text_count']} texts in {result['processing_time']:.2f}s")
            if result.get('sample_texts'):
                print(f"  Sample texts: {result['sample_texts']}")
            # Run actual extraction to get all texts/bboxes
            texts, bboxes, _ = ocr_engine.extract_text(test_image_path, output_format='xyxy')
            save_ocr_results(engine_name, texts, bboxes)
            ocr_results[engine_name] = {"texts": texts, "bboxes": bboxes}
        elif result.get("status") == "failed":
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"  - Not available")
    
    return ocr_results


def test_fallback_system(ocr_engine: OCREngine, test_image_path: str):
    """Test the fallback system"""
    print("\n" + "=" * 60)
    print("TESTING FALLBACK SYSTEM")
    print("=" * 60)
    
    try:
        start_time = time.time()
        texts, bboxes, metadata = ocr_engine.extract_text(
            test_image_path,
            output_format='xyxy',
            verbose=True
        )
        total_time = time.time() - start_time
        
        print(f"✓ OCR extraction successful!")
        print(f"  Engine used: {metadata['engine_name']}")
        print(f"  Fallback used: {metadata['fallback_used']}")
        print(f"  Processing time: {metadata['processing_time']:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Text count: {metadata['text_count']}")
        print(f"  Bbox count: {metadata['bbox_count']}")
        
        if texts:
            print(f"  Sample texts: {texts[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR extraction failed: {e}")
        return False


def test_backward_compatibility(test_image_path: str):
    """Test backward compatibility with existing code"""
    print("\n" + "=" * 60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)
    
    try:
        from util.utils import check_ocr_box
        
        # Test with EasyOCR (default)
        print("Testing with EasyOCR (default):")
        start_time = time.time()
        (texts, bboxes), _ = check_ocr_box(
            test_image_path,
            display_img=False,
            output_bb_format='xyxy',
            easyocr_args={'text_threshold': 0.9}
        )
        processing_time = time.time() - start_time
        
        print(f"  ✓ Success: {len(texts)} texts in {processing_time:.2f}s")
        if texts:
            print(f"  Sample texts: {texts[:3]}")
        
        # Test with PaddleOCR
        print("\nTesting with PaddleOCR:")
        start_time = time.time()
        (texts_paddle, bboxes_paddle), _ = check_ocr_box(
            test_image_path,
            display_img=False,
            output_bb_format='xyxy',
            use_paddleocr=True
        )
        processing_time = time.time() - start_time
        
        print(f"  ✓ Success: {len(texts_paddle)} texts in {processing_time:.2f}s")
        if texts_paddle:
            print(f"  Sample texts: {texts_paddle[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def test_global_ocr_engine(test_image_path: str):
    """Test the global OCR engine instance"""
    print("\n" + "=" * 60)
    print("TESTING GLOBAL OCR ENGINE")
    print("=" * 60)
    
    try:
        global_engine = get_global_ocr_engine()
        
        start_time = time.time()
        texts, bboxes, metadata = global_engine.extract_text(test_image_path)
        processing_time = time.time() - start_time
        
        print(f"✓ Global OCR engine test successful!")
        print(f"  Engine used: {metadata['engine_name']}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Text count: {len(texts)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Global OCR engine test failed: {e}")
        return False


def find_test_image():
    """Find a test image to use"""
    test_locations = [
        "./imgs/tsserrors/frame_0034.png",
        "./imgs/test.png",
        "./imgs/example.png", 
        "./test.png",
        "./example.png",
        "./sample.png",
        "./output/frame_0001/20250629_140822_frame_0001_original.png",
        "./output/frame_0002/20250629_140822_frame_0002_original.png",
        "./output/frame_0003/20250629_140822_frame_0003_original.png"
    ]
    
    for loc in test_locations:
        path = Path(loc)
        if path.exists():
            return str(path)
    
    return None


def preprocess_for_ocr(image):
    # 1. Upscale (2-4x)
    # 2. Convert to grayscale using best channel
    # 3. Apply CLAHE
    # 4. Adaptive thresholding
    # 5. Morphological cleanup
    # 6. Sharpening
    pass


def main():
    """Main test function"""
    print("OCR FALLBACK SYSTEM TEST")
    print("=" * 60)
    
    # Find test image
    test_image_path = find_test_image()
    if not test_image_path:
        print("❌ No test image found!")
        print("Please place a test image in one of these locations:")
        print("  - ./imgs/tsserrors/frame_0034.png")
        print("  - ./imgs/test.png")
        print("  - ./imgs/example.png")
        print("  - ./test.png")
        print("  - ./example.png")
        print("  - ./sample.png")
        return False
    
    print(f"Using test image: {test_image_path}")
    
    # Test 1: Engine initialization
    ocr_engine = test_ocr_engine_initialization()
    
    # Test 2: Individual engines (now returns ocr_results)
    ocr_results = test_ocr_engines_individually(ocr_engine, test_image_path)
    
    # Test 3: Fallback system
    fallback_success = test_fallback_system(ocr_engine, test_image_path)
    
    # Test 4: Backward compatibility
    compatibility_success = test_backward_compatibility(test_image_path)
    
    # Test 5: Global OCR engine
    global_success = test_global_ocr_engine(test_image_path)
    
    # Comparison Table
    if ocr_results:
        compare_ocr_results(ocr_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    available_engines = ocr_engine.get_available_engines()
    print(f"Available engines: {len(available_engines)}/{3}")
    for engine in ['easyocr', 'paddleocr', 'tesseract']:
        status = "✓" if engine in available_engines else "✗"
        print(f"  {status} {engine}")
    
    print(f"\nFallback system: {'✓' if fallback_success else '✗'}")
    print(f"Backward compatibility: {'✓' if compatibility_success else '✗'}")
    print(f"Global engine: {'✓' if global_success else '✗'}")
    
    if fallback_success and compatibility_success and global_success:
        print("\n🎉 ALL TESTS PASSED! OCR fallback system is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 