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

print(f"Detected {len(circles[0])} raw circles before filtering")

# Check grid spacing
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
x_sorted = np.sort(x_coords)
x_diffs = np.diff(x_sorted)
x_diffs = x_diffs[x_diffs > 10]
grid_spacing_x = np.median(x_diffs)
tolerance = grid_spacing_x * 0.3

print(f"Grid spacing X: {grid_spacing_x:.1f}")
print(f"Tolerance: {tolerance:.1f}")