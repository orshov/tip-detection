import cv2
import numpy as np
from src.image_utils import CIRCULARITY_THRESHOLD, AREA_RANGE

class TipDetector:
    def __init__(self):
        self.detected_tips = []
    
    def detect(self, image):
        """Detect tips in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        tips = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if AREA_RANGE[0] < area < AREA_RANGE[1]:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity > CIRCULARITY_THRESHOLD:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            tips.append((cx, cy, area))
        
        # Remove duplicates (within 5 pixels)
        filtered_tips = []
        for tip in tips:
            is_duplicate = False
            for existing_tip in filtered_tips:
                dist = np.sqrt((tip[0] - existing_tip[0])**2 + (tip[1] - existing_tip[1])**2)
                if dist < 5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_tips.append(tip)
        
        self.detected_tips = filtered_tips
        return filtered_tips
    
    def draw_results(self, image):
        """Draw detected tips on image"""
        result = image.copy()
        for tip in self.detected_tips:
            cv2.circle(result, (tip[0], tip[1]), 5, (0, 255, 0), 2)
        return result