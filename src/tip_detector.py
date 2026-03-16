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
        self.grid_spacing = None
    
    def find_grid_spacing(self, image):
        """Find the spacing between holes in the matrix"""
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
            
            # Calculate spacing from sorted coordinates
            x_sorted = np.sort(x_coords)
            y_sorted = np.sort(y_coords)
            
            # Find most common spacing (mode of differences)
            x_diffs = np.diff(x_sorted)
            y_diffs = np.diff(y_sorted)
            
            x_diffs = x_diffs[x_diffs > 10]  # Filter out small differences
            y_diffs = y_diffs[y_diffs > 10]
            
            if len(x_diffs) > 0 and len(y_diffs) > 0:
                self.grid_spacing = {
                    'x': np.median(x_diffs),
                    'y': np.median(y_diffs)
                }
            
            # Set matrix bounds
            padding = self.max_radius + 20
            self.matrix_bounds = {
                'min_x': int(x_coords.min()) - padding,
                'max_x': int(x_coords.max()) + padding,
                'min_y': int(y_coords.min()) - padding,
                'max_y': int(y_coords.max()) + padding
            }
    
    def snap_to_grid(self, x, y):
        """Placeholder - grid snapping disabled"""
        return x, y, True
    
    def is_in_matrix(self, x, y):
        """Check if point is within matrix bounds"""
        if self.matrix_bounds is None:
            return True
        
        return (self.matrix_bounds['min_x'] <= x <= self.matrix_bounds['max_x'] and
                self.matrix_bounds['min_y'] <= y <= self.matrix_bounds['max_y'])
    
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
                
                tips.append({
                    'x': int(x),
                    'y': int(y),
                    'radius': int(radius),
                    'area': np.pi * radius ** 2
                })
        
        # Remove duplicates (same position)
        unique_tips = []
        seen_positions = set()
        for tip in tips:
            pos_key = (tip['x'], tip['y'])
            if pos_key not in seen_positions:
                unique_tips.append(tip)
                seen_positions.add(pos_key)
        
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
            x = tip['x']
            y = tip['y']
            radius = tip['radius']
            
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