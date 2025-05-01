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

def apply_skeletonization(image, max_iterations=100):
    """Skeletonisasi citra dengan batas maksimum iterasi"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    skeleton = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    img = binary.copy()
    iterations = 0

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        iterations += 1
        if cv2.countNonZero(img) == 0 or iterations >= max_iterations:
            break

    print(f"Skeletonisasi selesai dalam {iterations} iterasi")
    return skeleton

def apply_opening(image, kernel_size=3):
    """Menerapkan operasi Opening (Erosi → Dilasi)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

def apply_closing(image, kernel_size=3):
    """Menerapkan operasi Closing (Dilasi → Erosi)"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed
