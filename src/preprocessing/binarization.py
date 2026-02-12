# src/preprocessing/binarization.py
import cv2

def adaptive_threshold(image):
    """
    Adaptive thresholding for uneven lighting
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )