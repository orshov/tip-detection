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
            minRadius=self.min_radius,
            maxRadius=self.max_radius
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
        
        # Snap tips to grid
        unique_tips = self.snap_to_grid(unique_tips)
        self.detected_tips = unique_tips
        
        return tips
    
    def snap_to_grid(self, tips):
        """Snap detections to grid positions based on hole clustering"""
        if not tips or len(tips) < 2:
            return tips
        
        # Group tips by approximate rows and columns
        tolerance = 35  # pixels
        
        # Sort by y then x
        tips_sorted = sorted(tips, key=lambda t: (t['y'], t['x']))
        
        # Group into rows
        rows = []
        current_row = []
        last_y = None
        
        for tip in tips_sorted:
            if last_y is None or abs(tip['y'] - last_y) < tolerance:
                current_row.append(tip)
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [tip]
            last_y = tip['y']
        
        if current_row:
            rows.append(current_row)
        
        # For each row, calculate average y and snap all tips in row
        snapped = []
        for row in rows:
            avg_y = np.mean([t['y'] for t in row])
            
            # Sort row by x
            row_sorted = sorted(row, key=lambda t: t['x'])
            
            # Group into columns
            cols = []
            current_col = []
            last_x = None
            
            for tip in row_sorted:
                if last_x is None or abs(tip['x'] - last_x) < tolerance:
                    current_col.append(tip)
                else:
                    if current_col:
                        cols.append(current_col)
                    current_col = [tip]
                last_x = tip['x']
            
            if current_col:
                cols.append(current_col)
            
            # For each column, calculate average x and snap
            for col in cols:
                avg_x = np.mean([t['x'] for t in col])
                
                # Take first tip from column as template
                template = col[0]
                snapped.append({
                    'x': avg_x,
                    'y': avg_y,
                    'radius': template['radius'],
                    'area': template['area']
                })
        
        return snapped
    
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