import cv2
from src.tip_detector import TipDetector

# Initialize detector with empty box reference
detector = TipDetector('data/empty_box.bmp')

# Load image with tips
image = cv2.imread('data/test_images/tips_sample.bmp')

if image is not None:
    tips = detector.detect(image)
    
    print(f"\nDetected {len(tips)} tips\n")
    for i, tip in enumerate(tips, 1):
        print(f"Tip {i}: Position ({tip['x']}, {tip['y']}), Radius: {tip['radius']}, Area: {tip['area']:.1f}, Circularity: {tip['circularity']:.2f}")
    
    # Draw and save
    result = detector.draw_results(image)
    cv2.imwrite('output/detection_result.jpg', result)
    print("\nResult saved to output/detection_result.jpg")
else:
    print("Image not found")