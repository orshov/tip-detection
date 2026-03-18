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
        self.grid_points = None
        self.grid_spacing = None
        self.hole_radius = None
        
        # Build grid from empty box
        self.build_grid_from_empty_box()
    
    def build_grid_from_empty_box(self):
        """Build a grid of expected hole positions from the empty box"""
        gray = cv2.cvtColor(self.empty_box, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect all holes in empty box
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
            
            # Store grid points and calculate average hole radius
            self.grid_points = circles[0, :, :2]  # x, y coordinates
            self.hole_radius = int(np.median(circles[0, :, 2]))
            
            # Calculate grid spacing
            x_coords = self.grid_points[:, 0]
            x_sorted = np.sort(x_coords)
            x_diffs = np.diff(x_sorted)
            x_diffs = x_diffs[x_diffs > 10]
            self.grid_spacing = np.median(x_diffs)
            
            # Set matrix bounds
            padding = self.max_radius + 20
            self.matrix_bounds = {
                'min_x': int(x_coords.min()) - padding,
                'max_x': int(x_coords.max()) + padding,
                'min_y': int(self.grid_points[:, 1].min()) - padding,
                'max_y': int(self.grid_points[:, 1].max()) + padding
            }
    
    def find_nearest_grid_point(self, x, y):
        """Find the nearest grid point and check if within tolerance"""
        if self.grid_points is None:
            return None, None, False
        
        distances = np.sqrt((self.grid_points[:, 0] - x)**2 + (self.grid_points[:, 1] - y)**2)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        tolerance = self.grid_spacing * 0.25  # 25% of grid spacing
        
        if min_distance < tolerance:
            grid_x, grid_y = self.grid_points[min_idx]
            return grid_x, grid_y, True
        
        return None, None, False
    
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
                
                # Snap to nearest grid point - only accept if close to grid
                grid_x, grid_y, is_valid = self.find_nearest_grid_point(x, y)
                
                if not is_valid:
                    continue  # Skip if not near a grid point (false positive)
                
                tips.append({
                    'x': float(grid_x),
                    'y': float(grid_y),
                    'radius': self.hole_radius,  # Use hole radius from empty box
                    'area': np.pi * self.hole_radius ** 2
                })
        
        # Remove duplicates (snap to grid already does this, but be safe)
        unique_tips = []
        seen_positions = set()
        for tip in tips:
            pos_key = (int(tip['x']), int(tip['y']))
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