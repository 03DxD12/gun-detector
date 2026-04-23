import json
import glob
import os
from pathlib import Path
import time
from app import detect_with_yolo, resolve_model_path, YOLO_MODEL, LABEL_ALIASES

# Force reloading the latest model
try:
    latest_model = resolve_model_path()
    print(f"Testing with model: {latest_model}")
except Exception as e:
    print(f"Error finding model: {e}")
    latest_model = None

def run_test_suite():
    test_images = glob.glob("test_suite/*.png")
    if not test_images:
        print("No test images found in test_suite/")
        return

    results = []
    passed = 0
    
    # Expected classes based on filename prefix
    # e.g., pistol_glock17.png -> PISTOL
    
    for img_path in test_images:
        filename = os.path.basename(img_path)
        expected_class = filename.split('_')[0].upper()
        
        print(f"Testing {filename} (Expected: {expected_class})...")
        
        try:
            # We call the detection logic directly
            # This will save a result image in static/uploads but we don't care
            predictions = detect_with_yolo(Path(img_path))
            
            # Check if expected class is in predictions
            best_conf = 0
            best_match = None
            for p in predictions:
                if p['class'] == expected_class:
                    if p['confidence'] > best_conf:
                        best_conf = p['confidence']
                        best_match = p
            
            status = "PASS" if best_conf >= 0.15 else "FAIL" # Low threshold for 2 epochs
            if status == "PASS":
                passed += 1
                
            results.append({
                "test_id": filename,
                "expected": expected_class,
                "detected": best_match['class'] if best_match else "NONE",
                "confidence": best_conf,
                "status": status
            })
            
            print(f"  Result: {status} (Conf: {best_conf})")
        except Exception as e:
            print(f"  Error testing {filename}: {e}")
            results.append({
                "test_id": filename,
                "status": "ERROR",
                "reason": str(e)
            })

    # Summary
    print("\n--- TEST SUMMARY ---")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {passed/len(results)*100:.1f}%")
    
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full report saved to test_results.json")

if __name__ == "__main__":
    run_test_suite()
