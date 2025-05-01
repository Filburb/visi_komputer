import cv2
import numpy as np

def normalize_image(image):
    """
    Normalisasi citra: skala nilai pixel ke rentang 0–255 (kontras disesuaikan).
    Cocok untuk pemrosesan visual.
    """
    norm_img = np.zeros_like(image)
    normalized = cv2.normalize(image, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized
