import cv2
import numpy as np

class TipDetector:
    def __init__(self, empty_box_path):
        """Initialize with reference empty box image"""
        self.empty_box = cv2.imread(empty_box_path)
        self.detected_tips = []
    
    def detect(self, image):
        """Detect tips using Hough Circle Detection on difference image"""
        # Convert to grayscale
        gray_tips = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_empty = cv2.cvtColor(self.empty_box, cv2.COLOR_BGR2GRAY)
        
        # Compute difference - tips will be brighter
        diff = gray_tips.astype(float) - gray_empty.astype(float)
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur for circle detection
        blurred = cv2.GaussianBlur(diff, (9, 9), 2)
        
        # Detect circles using Hough Circle Detection
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=30,
            param2=20,
            minRadius=15,
            maxRadius=35
        )
        
        tips = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                tips.append({
                    'x': int(x),
                    'y': int(y),
                    'radius': int(radius),
                    'area': np.pi * radius ** 2
                })
        
        self.detected_tips = tips
        return tips
    
    def draw_results(self, image):
        """Draw detected tips on image"""
        result = image.copy()
        for tip in self.detected_tips:
            x = tip['x']
            y = tip['y']
            radius = tip['radius']
            
            # Draw circle outline
            cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
        return result