# src/preprocessing/noise_removal.py
import cv2

def remove_noise(image):
    """
    Apply Gaussian Blur to reduce noise
    """
    return cv2.GaussianBlur(image, (5, 5), 0)