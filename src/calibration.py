import cv2
import numpy as np

class Calibration:
    def __init__(self):
        self.reference_pixel_size = 1.0
    
    def calibrate(self, image, reference_object_size):
        """Calibrate the pixel-to-mm conversion"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            self.reference_pixel_size = np.sqrt(area) / reference_object_size
            return self.reference_pixel_size
        return None