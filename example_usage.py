import cv2
from src.tip_detector import TipDetector

# Test 1: Small holes box
print("=" * 50)
print("Testing SMALL holes box")
print("=" * 50)

detector1 = TipDetector('data/empty_box.bmp', min_radius=15, max_radius=35)
image1 = cv2.imread('data/test_images/tips_sample.bmp')

if image1 is not None:
    tips1 = detector1.detect(image1)
    
    print(f"\nDetected {len(tips1)} tips\n")
    for i, tip in enumerate(tips1, 1):
        print(f"Tip {i}: Position ({tip['x']}, {tip['y']}), Radius: {tip['radius']}, Area: {tip['area']:.1f}")
    
    filepath1 = detector1.save_result(image1, prefix='small_holes')
    print(f"\nResult saved to {filepath1}")
else:
    print("Image not found")

# Test 2: Large holes box
print("\n" + "=" * 50)
print("Testing LARGE holes box")
print("=" * 50)

detector2 = TipDetector('data/large_holes_empty.bmp', min_radius=25, max_radius=50)
image2 = cv2.imread('data/test_images/large_holes_tips.bmp')

if image2 is not None:
    tips2 = detector2.detect(image2)
    
    print(f"\nDetected {len(tips2)} tips\n")
    for i, tip in enumerate(tips2, 1):
        print(f"Tip {i}: Position ({tip['x']}, {tip['y']}), Radius: {tip['radius']}, Area: {tip['area']:.1f}")
    
    filepath2 = detector2.save_result(image2, prefix='large_holes')
    print(f"\nResult saved to {filepath2}")
else:
    print("Image not found")