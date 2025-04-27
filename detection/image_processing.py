import cv2
import numpy as np
import csv

def read_image(image_path):
    """Membaca gambar dari path"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Tidak bisa membaca gambar: {image_path}")
    return img

def extract_15x15_matrix(image):
    """Mengambil matrix 15x15 piksel dari tengah gambar"""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    half = 15 // 2

    start_x = center_x - half
    start_y = center_y - half

    cropped = image[start_y:start_y+15, start_x:start_x+15]
    return cropped

def save_image(image, save_path):
    """Menyimpan gambar hasil enhancement ke disk"""
    # Pastikan format uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    cv2.imwrite(save_path, image)

def save_15x15_matrix_csv(matrix, save_path):
    """Menyimpan matrix 15x15 ke file CSV dalam skala 0-1"""
    if len(matrix.shape) == 3:
        matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)

    matrix_normalized = matrix / 255.0

    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in matrix_normalized:
            writer.writerow(["{:.4f}".format(val) for val in row])
