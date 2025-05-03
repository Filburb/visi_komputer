import cv2
import numpy as np

def apply_dilation(img):
    # Pastikan gambar sudah biner
    ret, thresholded_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)  # Kernel lebih besar untuk dilasi
    dilated_img = cv2.dilate(thresholded_img, kernel, iterations=1)  # Operasi dilasi
    return dilated_img

def apply_closing(img):
    # Pastikan gambar sudah biner
    ret, thresholded_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Gunakan kernel lebih besar untuk operasi closing
    kernel = np.ones((5, 5), np.uint8)  # Kernel lebih besar untuk closing
    closed_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)  # Operasi closing
    return closed_img


def apply_skeletonization(img):
    # Pastikan gambar biner
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)  # Operasi erosi
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:  # Jika tidak ada piksel putih
            break

    return skel
