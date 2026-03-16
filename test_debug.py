import cv2
import numpy as np

gray_tips = cv2.cvtColor(cv2.imread('data/test_images/tips_sample.bmp'), cv2.COLOR_BGR2GRAY)
gray_empty = cv2.cvtColor(cv2.imread('data/empty_box.bmp'), cv2.COLOR_BGR2GRAY)

diff = gray_tips.astype(float) - gray_empty.astype(float)
diff = np.clip(diff, 0, 255).astype(np.uint8)

_, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

good_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if 200 < area < 500:
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            good_contours.append({'area': area, 'circularity': circularity})

print(f"Found {len(good_contours)} good contours (area 200-500)")
if good_contours:
    circularities = [c['circularity'] for c in good_contours]
    print(f"Circularity min: {min(circularities):.2f}, max: {max(circularities):.2f}, mean: {np.mean(circularities):.2f}")