import cv2
import numpy as np
from datetime import datetime
import os

class TipDetector:
    def __init__(self, empty_box_path, min_radius=15, max_radius=35):
        """Initialize with reference empty box image"""
        self.empty_box = cv2.imread(empty_box_path)
        self.detected_tips = []
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.matrix_bounds = None
        self.hole_radius = None
        
        # Detect hole radius from empty box
        self.detect_hole_radius()
    
    def detect_hole_radius(self):
        """Detect the actual hole radius from the empty box"""
        gray = cv2.cvtColor(self.empty_box, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=30,
            param2=15,
            minRadius=self.min_radius - 5,
            maxRadius=self.max_radius + 5
        )
        
        if circles is not None:
            # Use median radius from all detected holes
            self.hole_radius = int(np.median(circles[0, :, 2]))
    
    def find_matrix_bounds(self, image):
        """Find the boundaries of the tip matrix by detecting the grid"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find all circles (both empty and with tips)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=30,
            param2=15,
            minRadius=self.min_radius - 5,
            maxRadius=self.max_radius + 5
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x_coords = circles[0, :, 0]
            y_coords = circles[0, :, 1]
            
            # Add padding
            padding = self.max_radius + 20
            self.matrix_bounds = {
                'min_x': int(x_coords.min()) - padding,
                'max_x': int(x_coords.max()) + padding,
                'min_y': int(y_coords.min()) - padding,
                'max_y': int(y_coords.max()) + padding
            }
    
    def is_in_matrix(self, x, y):
        """Check if point is within matrix bounds"""
        if self.matrix_bounds is None:
            return True
        
        return (self.matrix_bounds['min_x'] <= x <= self.matrix_bounds['max_x'] and
                self.matrix_bounds['min_y'] <= y <= self.matrix_bounds['max_y'])
    
    def refine_center_with_brightness(self, diff_image, x, y, radius):
        """Refine circle center by finding the brightest point in the region"""
        margin = int(radius * 1.5)
        x1 = max(0, int(x - margin))
        y1 = max(0, int(y - margin))
        x2 = min(diff_image.shape[1], int(x + margin))
        y2 = min(diff_image.shape[0], int(y + margin))
        
        roi = diff_image[y1:y2, x1:x2]
        
        if roi.size == 0 or roi.max() == 0:
            return x, y
        
        # Find the brightest point (tip center) in the ROI
        y_local, x_local = np.unravel_index(roi.argmax(), roi.shape)
        refined_x = x1 + x_local
        refined_y = y1 + y_local
        
        return refined_x, refined_y
    
    def is_likely_false_positive(self, diff_image, x, y, radius):
        """Check if this detection is likely a false positive (between holes)"""
        # Extract small region around center
        margin = int(radius * 0.8)
        x1 = max(0, int(x - margin))
        y1 = max(0, int(y - margin))
        x2 = min(diff_image.shape[1], int(x + margin))
        y2 = min(diff_image.shape[0], int(y + margin))
        
        roi = diff_image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return True
        
        # False positives have very low signal
        center_value = roi[len(roi)//2, len(roi[0])//2]
        roi_max = roi.max()
        
        # If center is much weaker than max in region, likely false positive
        if center_value < roi_max * 0.3:
            return True
        
        return False

    def detect(self, image):
        """Detect tips using Hough Circle Detection on difference image"""
        # Convert to grayscale
        gray_tips = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_empty = cv2.cvtColor(self.empty_box, cv2.COLOR_BGR2GRAY)
        
        # Find matrix bounds from the empty box
        self.find_matrix_bounds(self.empty_box)
        
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
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        tips = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                
                # Filter: only keep circles within matrix bounds
                if not self.is_in_matrix(x, y):
                    continue
                
                # Filter: reject likely false positives
                if self.is_likely_false_positive(diff, x, y, radius):
                    continue
                
                # Refine center position using brightness
                refined_x, refined_y = self.refine_center_with_brightness(diff, x, y, radius)
                
                tips.append({
                    'x': int(refined_x),
                    'y': int(refined_y),
                    'radius': self.hole_radius if self.hole_radius else int(radius),
                    'area': np.pi * (self.hole_radius if self.hole_radius else int(radius)) ** 2
                })
        
        # Remove duplicates (same position within 10 pixels)
        unique_tips = []
        seen_positions = []
        for tip in tips:
            pos = (tip['x'], tip['y'])
            
            # Check if position is too close to existing tip
            is_duplicate = False
            for seen_pos in seen_positions:
                distance = np.sqrt((pos[0] - seen_pos[0])**2 + (pos[1] - seen_pos[1])**2)
                if distance < 10:  # Within 10 pixels = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tips.append(tip)
                seen_positions.append(pos)
        
        self.detected_tips = unique_tips
        return unique_tips
    
    def draw_results(self, image):
        """Draw detected tips on image"""
        result = image.copy()
        
        # Draw matrix bounds if available
        if self.matrix_bounds:
            cv2.rectangle(
                result,
                (self.matrix_bounds['min_x'], self.matrix_bounds['min_y']),
                (self.matrix_bounds['max_x'], self.matrix_bounds['max_y']),
                (255, 0, 0), 2
            )
        
        # Draw detected tips
        for tip in self.detected_tips:
            x = int(tip['x'])
            y = int(tip['y'])
            radius = int(tip['radius'])
            
            # Draw circle outline
            cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
        
        return result
    
    def save_result(self, image, output_dir='output', prefix='detection'):
        """Save detection result with unique timestamp filename"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        result = self.draw_results(image)
        cv2.imwrite(filepath, result)
        
        return filepath