import cv2
import numpy as np
from datetime import datetime
import os

class TipDetector:
    def __init__(self, empty_box_path):
        """Initialize with reference empty box image"""
        self.empty_box = cv2.imread(empty_box_path)
        self.detected_tips = []
        self.matrix_bounds = None
        self.hole_positions = None
        
    def find_hole_grid(self, image):
        """Detect the grid of holes and their positions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=30,
            param2=15,
            minRadius=10,
            maxRadius=40
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            self.hole_positions = circles[0, :]
            
            x_coords = circles[0, :, 0]
            y_coords = circles[0, :, 1]
            
            padding = 50
            self.matrix_bounds = {
                'min_x': int(x_coords.min()) - padding,
                'max_x': int(x_coords.max()) + padding,
                'min_y': int(y_coords.min()) - padding,
                'max_y': int(y_coords.max()) + padding
            }

    def snap_to_grid(self, tips):
        """Snap detections to the nearest grid positions using row/column averaging"""
        if not tips or self.hole_positions is None:
            return tips
        
        # Group hole positions by approximate row/column
        hole_rows = {}
        hole_cols = {}
        
        tolerance = 30  # pixels - tips within this distance belong to same row/col
        
        # Cluster holes into rows and columns
        for hole in self.hole_positions:
            x, y, r = hole
            
            # Find row
            found_row = False
            for row_y in list(hole_rows.keys()):
                if abs(y - row_y) < tolerance:
                    hole_rows[row_y].append(x)
                    found_row = True
                    break
            if not found_row:
                hole_rows[y] = [x]
            
            # Find column
            found_col = False
            for col_x in list(hole_cols.keys()):
                if abs(x - col_x) < tolerance:
                    hole_cols[col_x].append(y)
                    found_col = True
                    break
            if not found_col:
                hole_cols[x] = [y]
        
        # Calculate average row and column positions
        avg_rows = {row_y: np.mean(xs) for row_y, xs in hole_rows.items()}
        avg_cols = {col_x: np.mean(ys) for col_x, ys in hole_cols.items()}
        
        # Snap tips to nearest grid positions
        snapped_tips = []
        for tip in tips:
            tx, ty = tip['x'], tip['y']
            
            # Find nearest row and column
            nearest_row = min(avg_rows.keys(), key=lambda ry: abs(ty - ry))
            nearest_col = min(avg_cols.keys(), key=lambda cx: abs(tx - cx))
            
            snapped_x = avg_cols[nearest_col]
            snapped_y = avg_rows[nearest_row]
            
            snapped_tips.append({
                'x': snapped_x,
                'y': snapped_y,
                'radius': tip.get('radius', 25),
                'area': tip.get('area', 0)
            })
        
        return snapped_tips

    def detect(self, image):
        """Detect tips using brightness difference"""
        gray_tips = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_empty = cv2.cvtColor(self.empty_box, cv2.COLOR_BGR2GRAY)
        
        # Find hole grid from empty box
        self.find_hole_grid(self.empty_box)
        
        # Compute difference
        diff = gray_tips.astype(float) - gray_empty.astype(float)
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        blurred = cv2.GaussianBlur(diff, (9, 9), 2)
        
        # Detect circles
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
        
        tips = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                
                # Only keep tips within matrix bounds
                if self.matrix_bounds:
                    if not (self.matrix_bounds['min_x'] <= x <= self.matrix_bounds['max_x'] and
                            self.matrix_bounds['min_y'] <= y <= self.matrix_bounds['max_y']):
                        continue
                
                tips.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': 25,
                    'area': np.pi * 25 ** 2
                })
        
        # Snap to grid
        tips = self.snap_to_grid(tips)
        
        # Remove duplicates
        unique_tips = []
        seen = set()
        for tip in tips:
            pos = (round(tip['x']), round(tip['y']))
            if pos not in seen:
                unique_tips.append(tip)
                seen.add(pos)
        
        self.detected_tips = unique_tips
        return unique_tips
    
    def draw_results(self, image):
        """Draw detected tips on image"""
        result = image.copy()
        
        for tip in self.detected_tips:
            x = int(tip['x'])
            y = int(tip['y'])
            radius = int(tip['radius'])
            
            cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
        
        return result
    
    def save_result(self, image, output_dir='output', prefix='detection'):
        """Save detection result"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        result = self.draw_results(image)
        cv2.imwrite(filepath, result)
        
        return filepath