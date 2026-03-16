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

# Get grid points from empty box
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

grid_points = circles_empty[0, :]
x_sorted = np.sort(circles_empty[0, :, 0])
x_diffs = np.diff(x_sorted)
x_diffs = x_diffs[x_diffs > 10]
grid_spacing_x = np.median(x_diffs)
tolerance = grid_spacing_x * 0.4

print(f"Grid points: {len(grid_points)}")
print(f"Tolerance: {tolerance:.1f}")

# Check alignment for each detected circle
rejected = []
for i, circle in enumerate(circles[0]):
    x, y, radius = circle
    distances = np.sqrt((grid_points[:, 0] - x)**2 + (grid_points[:, 1] - y)**2)
    min_distance = distances.min()
    
    if min_distance >= tolerance:
        rejected.append({
            'index': i,
            'x': x,
            'y': y,
            'distance': min_distance
        })

print(f"\nRejected circles (distance >= tolerance):")
for r in rejected:
    print(f"  Circle {r['index']}: ({r['x']}, {r['y']}) - distance: {r['distance']:.1f}")