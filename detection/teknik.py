import cv2
import numpy as np

def apply_edge_detection(image):
    """Menerapkan Edge Detection (Canny)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

def apply_harris_corner_detection(image):
    """Menerapkan Harris Corner Detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # Normalisasi hasil agar bisa divisualisasikan
    dst = cv2.dilate(dst, None)
    image_result = np.copy(image)
    image_result[dst > 0.01 * dst.max()] = [0, 0, 255]  # tanda sudut warna merah
    return image_result
