# src/preprocessing/pipeline.py
import cv2
from src.preprocessing.grayscale import to_grayscale
from src.preprocessing.noise_removal import remove_noise
from src.preprocessing.binarization import adaptive_threshold
from src.preprocessing.skew_correction import correct_skew

def preprocess_image(image_path):
    """
    Full preprocessing pipeline
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at: {image_path}")

    gray = to_grayscale(image)
    denoised = remove_noise(gray)
    binary = adaptive_threshold(denoised)
    corrected = correct_skew(binary)

    return corrected