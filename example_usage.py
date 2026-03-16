import cv2
from src.tip_detector import TipDetector

# Load image
image = cv2.imread('data/test_images/tips_sample.bmp')

if image is not None:
    detector = TipDetector()
    tips = detector.detect(image)
    
    print(f"Detected {len(tips)} tips")
    for i, tip in enumerate(tips):
        print(f"Tip {i+1}: Position ({tip[0]}, {tip[1]}), Area: {tip[2]}")
    
    # Draw and save
    result = detector.draw_results(image)
    cv2.imwrite('output/detection_result.jpg', result)
    print("Result saved to output/detection_result.jpg")
else:
    print("Image not found")