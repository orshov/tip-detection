import cv2
import numpy as np

gray_tips = cv2.cvtColor(cv2.imread('data/test_images/tips_sample.bmp'), cv2.COLOR_BGR2GRAY)
gray_empty = cv2.cvtColor(cv2.imread('data/empty_box.bmp'), cv2.COLOR_BGR2GRAY)

diff = gray_tips.astype(float) - gray_empty.astype(float)
diff = np.clip(diff, 0, 255).astype(np.uint8)

blurred = cv2.GaussianBlur(diff, (9, 9), 2)

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

print(f"Total detected circles: {len(circles[0])}")

# Get matrix bounds
gray = cv2.cvtColor(cv2.imread('data/empty_box.bmp'), cv2.COLOR_BGR2GRAY)
blurred_empty = cv2.GaussianBlur(gray, (9, 9), 2)
circles_empty = cv2.HoughCircles(
    blurred_empty,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=40,
    param1=30,
    param2=15,
    minRadius=10,
    maxRadius=40
)

x_coords = circles_empty[0, :, 0]
y_coords = circles_empty[0, :, 1]

min_radius = 15
max_radius = 35
padding = max_radius + 20

matrix_bounds = {
    'min_x': int(x_coords.min()) - padding,
    'max_x': int(x_coords.max()) + padding,
    'min_y': int(y_coords.min()) - padding,
    'max_y': int(y_coords.max()) + padding
}

print(f"\nMatrix bounds:")
print(f"  X: {matrix_bounds['min_x']} to {matrix_bounds['max_x']}")
print(f"  Y: {matrix_bounds['min_y']} to {matrix_bounds['max_y']}")

# Check each circle
out_of_bounds = []
for i, circle in enumerate(circles[0]):
    x, y, radius = circle
    in_bounds = (matrix_bounds['min_x'] <= x <= matrix_bounds['max_x'] and
                 matrix_bounds['min_y'] <= y <= matrix_bounds['max_y'])
    
    if not in_bounds:
        out_of_bounds.append({
            'index': i,
            'x': x,
            'y': y,
            'reason': f"X out" if not (matrix_bounds['min_x'] <= x <= matrix_bounds['max_x']) else "Y out"
        })

print(f"\nOut of bounds circles: {len(out_of_bounds)}")
for o in out_of_bounds:
    print(f"  Circle {o['index']}: ({o['x']}, {o['y']}) - {o['reason']}")