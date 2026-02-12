# src/preprocessing/grayscale.py
import cv2

def to_grayscale(image):
    """
    Convert BGR image to grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)