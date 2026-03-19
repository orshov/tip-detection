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
        self.hole_positions = []
    
    def find_hole_grid(self, image):
        """Detect hole positions in the empty box"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=30,
            param2=15,
            minRadius=15,
            maxRadius=35
        )
        
        self.hole_positions = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            self.hole_positions = [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0, :]]
        
        return self.hole_positions

    def is_tip_in_hole(self, tip_x, tip_y, threshold=50):
        """Check if a detected tip is close to an actual hole position"""
        if not self.hole_positions:
            return True
        
        # Find closest hole
        min_distance = float('inf')
        for hole_x, hole_y, hole_r in self.hole_positions:
            distance = np.sqrt((tip_x - hole_x)**2 + (tip_y - hole_y)**2)
            min_distance = min(min_distance, distance)
        
        # If within threshold of a hole, it's valid
        return min_distance < threshold
    
    def snap_to_grid(self, tips):
        """Snap detections to grid positions based on actual grid spacing"""
        if not tips or len(tips) < 3:
            return tips
        
        # Find the actual grid spacing by analyzing detected positions
        xs = sorted(set([int(t['x']) for t in tips]))
        ys = sorted(set([int(t['y']) for t in tips]))
        
        if len(xs) < 2 or len(ys) < 2:
            return tips
        
        # Calculate average spacing
        x_diffs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
        y_diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
        
        x_spacing = int(np.median(x_diffs))
        y_spacing = int(np.median(y_diffs))
        
        # Snap each tip to nearest grid point
        snapped = []
        seen = set()
        
        for tip in tips:
            # Find nearest grid line
            nearest_x = round(tip['x'] / x_spacing) * x_spacing
            nearest_y = round(tip['y'] / y_spacing) * y_spacing
            
            # Avoid duplicates
            pos = (nearest_x, nearest_y)
            if pos not in seen:
                snapped.append({
                    'x': float(nearest_x),
                    'y': float(nearest_y),
                    'radius': tip['radius'],
                    'area': tip['area']
                })
                seen.add(pos)
        
        return snapped

    def detect(self, image):
        """Detect tips using Hough Circle Detection on difference image"""
        # Find hole grid from empty box first
        self.find_hole_grid(self.empty_box)
        
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
                
                # Filter: only keep tips that are close to actual holes
                if self.is_tip_in_hole(x, y, threshold=60):
                    tips.append({
                        'x': int(x),
                        'y': int(y),
                        'radius': int(radius),
                        'area': np.pi * radius ** 2
                    })
        
        # Remove duplicates
        unique_tips = []
        seen = set()
        for tip in tips:
            pos = (round(tip['x']), round(tip['y']))
            if pos not in seen:
                unique_tips.append(tip)
                seen.add(pos)
        
        # Snap tips to grid
        snapped_tips = self.snap_to_grid(unique_tips)
        self.detected_tips = snapped_tips
        
        return snapped_tips
    
    def draw_results(self, image):
        """Draw detected tips on image"""
        result = image.copy()
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