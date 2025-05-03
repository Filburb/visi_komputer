import cv2
import numpy as np
import csv

def detect_keypoints_orb(img):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    keypoints_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
    return keypoints, descriptors, keypoints_img

def match_features_orb(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:20]  # ambil 20 terbaik

def draw_matches(img1, kp1, img2, kp2, matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

def extract_15x15_matrix(img):
    """
    Ambil cuplikan matrix 15x15 dari tengah gambar.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    half = 7  # setengah dari 15 (7) agar total 15x15
    return gray[cy - half:cy + half + 1, cx - half:cx + half + 1]

def save_15x15_matrix_csv(matrix, path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix:
            writer.writerow(row)
