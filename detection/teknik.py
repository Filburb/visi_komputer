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

    dst = cv2.dilate(dst, None)
    image_result = np.copy(image)
    image_result[dst > 0.01 * dst.max()] = [0, 0, 255]  # sudut warna merah
    return image_result

def apply_sift_detection(image):
    """Deteksi fitur menggunakan SIFT (untuk visualisasi, tidak matching)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    output = cv2.drawKeypoints(image, keypoints, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return output
