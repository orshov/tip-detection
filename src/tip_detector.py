import cv2
import numpy as np

class TipDetector:
    def __init__(self, empty_box_path):
        """Initialize with reference empty box image"""
        self.empty_box = cv2.imread(empty_box_path)
        self.detected_tips = []
    
    def detect(self, image):
        """Detect tips by analyzing brightness differences and convexity"""
        # Convert to grayscale
        gray_tips = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_empty = cv2.cvtColor(self.empty_box, cv2.COLOR_BGR2GRAY)
        
        # Compute difference - tips will be brighter
        diff = gray_tips.astype(float) - gray_empty.astype(float)
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        # Apply threshold - only keep bright differences (tips are lighter)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size - tips should be similar size to holes
            if 100 < area < 600:
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Good circles with high circularity
                    if circularity > 0.6:
                        tips.append({
                            'x': int(x),
                            'y': int(y),
                            'radius': int(radius),
                            'area': area,
                            'circularity': circularity
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