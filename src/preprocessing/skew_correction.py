import cv2
import numpy as np

def correct_skew(image):
    """
    Improved skew correction using Hough Transform
    Works better for handwritten prescriptions
    """

    # Step 1: Edge detection
    edges = cv2.Canny(image, 50, 150)

    # Step 2: Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return image  # fallback if no lines detected

    angles = []

    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    # Step 3: Use median angle (robust)
    median_angle = np.median(angles)

    # Step 4: Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated