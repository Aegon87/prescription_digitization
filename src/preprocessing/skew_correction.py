import cv2
import numpy as np

# def correct_skew(image):
#     """
#     Improved skew correction using Hough Transform
#     Works better for handwritten prescriptions
#     """

#     # Step 1: Edge detection
#     edges = cv2.Canny(image, 50, 150)

#     # Step 2: Detect lines
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

#     if lines is None:
#         return image  # fallback if no lines detected

#     angles = []

#     for rho, theta in lines[:, 0]:
#         angle = (theta * 180 / np.pi) - 90
#         angles.append(angle)

#     # Step 3: Use median angle (robust)
#     median_angle = np.median(angles)

#     # Step 4: Rotate image
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)

#     M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

#     rotated = cv2.warpAffine(
#         image,
#         M,
#         (w, h),
#         flags=cv2.INTER_CUBIC,
#         borderMode=cv2.BORDER_REPLICATE
#     )

#     return rotated

def correct_skew(image):
    """
    Skew correction for already preprocessed (thresholded) image
    """

    # Morphological closing to connect text
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image  # safety fallback

    # Largest contour (main text region)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get angle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    # Rotate original image (NOT processed)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated