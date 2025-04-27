import cv2
import numpy as np

def apply_dilation(image, kernel_size=3, iterations=1):
    """Menerapkan Dilasi pada citra"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated

def apply_erosion(image, kernel_size=3, iterations=1):
    """Menerapkan Erosi pada citra"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded

def apply_skeletonization(image):
    """Skeletonisasi citra"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold untuk binarisasi
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = np.zeros(binary.shape, np.uint8)
    eroded = np.zeros(binary.shape, np.uint8)

    img = binary.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skeleton
