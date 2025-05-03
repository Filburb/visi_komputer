import os
import cv2
import random
import numpy as np

from detection.feature import (
    extract_15x15_matrix,
    save_15x15_matrix_csv,
    detect_keypoints_orb,
    match_features_orb,
    draw_matches
)

from detection.teknik import (
    apply_edge_detection,
    apply_harris_corner_detection,
    apply_sift_detection
)

from detection.image_preprocessing import normalize_image
from detection.image_processing import (
    read_image,
    save_image
)

from detection.konvolusi import (
    apply_gaussian_blur,
    apply_sobel_filter,
    apply_prewitt_filter
)

from detection.morfologi import (
    apply_dilation,
    apply_closing,  # Ganti dari apply_opening ke apply_closing
    apply_skeletonization
)

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def main():
    input_folder = "00000"
    output_folder = "output"
    ensure_folder(output_folder)

    files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if len(files) < 2:
        print("Minimal 2 gambar diperlukan untuk matching.")
        return

    skeleton_images = {}  # simpan hasil skeleton untuk ORB matching

    for index, file in enumerate(files):
        img_path = os.path.join(input_folder, file)
        img = read_image(img_path)

        # 1. Normalisasi
        norm_img = normalize_image(img)
        save_image(norm_img, os.path.join(output_folder, f"normalized_{file}"))
        print(f"[{index+1}] Normalized: {file}")

        # 2. Edge Detection + SIFT
        edge_img = apply_edge_detection(norm_img)
        save_image(edge_img, os.path.join(output_folder, f"canny_{file}"))

        sift_img = apply_sift_detection(norm_img)
        save_image(sift_img, os.path.join(output_folder, f"sift_{file}"))

        # 3. Harris Corner Detection (Tambah di sini)
        harris_img = apply_harris_corner_detection(norm_img)
        save_image(harris_img, os.path.join(output_folder, f"harris_{file}"))

        # 4. Konvolusi
        gaussian_img = apply_gaussian_blur(edge_img)
        save_image(gaussian_img, os.path.join(output_folder, f"gaussian_{file}"))

        sobel_img = apply_sobel_filter(edge_img)
        save_image(sobel_img, os.path.join(output_folder, f"sobel_{file}"))

        prewitt_img = apply_prewitt_filter(edge_img)
        save_image(prewitt_img, os.path.join(output_folder, f"prewitt_{file}"))

        # 5. Morfologi - Dilasi dan Closing
        dilated_img = apply_dilation(edge_img)
        save_image(dilated_img, os.path.join(output_folder, f"dilated_{file}"))

        # Menggunakan Closing
        closed_img = apply_closing(edge_img)  # Ganti opening ke closing
        save_image(closed_img, os.path.join(output_folder, f"closed_{file}"))

        # 6. Skeletonisasi
        skeleton_img = apply_skeletonization(edge_img)
        save_image(skeleton_img, os.path.join(output_folder, f"skeleton_{file}"))
        skeleton_images[file] = skeleton_img  # disimpan untuk ORB nanti

        # 7. ORB Keypoints dari hasil skeletonisasi
        _, _, kp_img = detect_keypoints_orb(skeleton_img)
        save_image(kp_img, os.path.join(output_folder, f"keypoints_{file}"))

        # 8. Cuplikan matrix 15x15 (gambar pertama saja)
        if index == 0:
            print(f"➡️ Ambil matrix 15x15 dari: {file}")
            matrix = extract_15x15_matrix(norm_img)
            save_15x15_matrix_csv(matrix, os.path.join(output_folder, "matrix15x15.csv"))

            save_image(apply_edge_detection(matrix), os.path.join(output_folder, "canny_matrix15x15.jpg"))
            save_image(apply_sift_detection(matrix), os.path.join(output_folder, "sift_matrix15x15.jpg"))
            save_image(apply_gaussian_blur(matrix), os.path.join(output_folder, "gaussian_matrix15x15.jpg"))
            save_image(apply_sobel_filter(matrix), os.path.join(output_folder, "sobel_matrix15x15.jpg"))
            save_image(apply_prewitt_filter(matrix), os.path.join(output_folder, "prewitt_matrix15x15.jpg"))
            save_image(apply_dilation(matrix), os.path.join(output_folder, "dilated_matrix15x15.jpg"))
            save_image(apply_closing(matrix), os.path.join(output_folder, "closed_matrix15x15.jpg"))
            skeleton_matrix = apply_skeletonization(matrix)
            save_image(skeleton_matrix, os.path.join(output_folder, "skeleton_matrix15x15.jpg"))
            _, _, kp_matrix = detect_keypoints_orb(skeleton_matrix)
            save_image(kp_matrix, os.path.join(output_folder, "keypoints_matrix15x15.jpg"))

    # 9. Matching antara 2 gambar acak hasil skeletonisasi
    random_files = random.sample(files, 2)
    print(f"🔀 Matching: {random_files[0]} vs {random_files[1]}")

    img1 = skeleton_images[random_files[0]]
    img2 = skeleton_images[random_files[1]]

    kp1, desc1, _ = detect_keypoints_orb(img1)
    kp2, desc2, _ = detect_keypoints_orb(img2)

    if desc1 is None or desc2 is None:
        print("❌ Tidak ada deskriptor. Matching gagal.")
        return

    matches = match_features_orb(desc1, desc2)
    matched_img = draw_matches(img1, kp1, img2, kp2, matches)

    match_filename = f"matched_{random_files[0][:-4]}_vs_{random_files[1][:-4]}.jpg"
    save_image(matched_img, os.path.join(output_folder, match_filename))
    print(f"✅ Matching disimpan: {match_filename}")

if __name__ == "__main__":
    main()
