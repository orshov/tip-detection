def detect(self, image):
    # Assuming image is a properly sized matrix
    height, width = image.shape

    # Check matrix bounds
    if height < 2 or width < 2:
        return None  # or some indication of failure

    # Center contrast check (replace logic with actual implementation)
    center_value = image[height // 2, width // 2]
    contrast_threshold = 10  # Example threshold

    # Check for sufficient contrast
    if np.abs(center_value - np.mean(image)) < contrast_threshold:
        return None  # Insufficient contrast

    # (Other existing logic remains unchanged)
    # ...