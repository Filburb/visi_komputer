import os
import cv2
import random
from detection.image_processing import (
    read_image,
    extract_15x15_matrix,
    save_image,
    save_15x15_matrix_csv
)
from detection.image_preprocessing import normalize_image
from detection.teknik import (
    apply_edge_detection,
    apply_harris_corner_detection,
    apply_sift_detection
)
from detection.konvolusi import (
    apply_gaussian_blur,
    apply_sobel_filter,
    apply_prewitt_filter
)
from detection.morfologi import (
    apply_dilation,
    apply_erosion,
    apply_skeletonization,
    apply_opening,
    apply_closing
)
from detection.feature import (
    detect_keypoints_orb,
    match_features_orb,
    draw_matches
)

def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

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

    for index, file in enumerate(files):
        img_path = os.path.join(input_folder, file)
        img = read_image(img_path)

        # 1. Normalisasi
        norm_img = normalize_image(img)
        save_image(norm_img, os.path.join(output_folder, f"normalized_{file}"))
        print(f"[{index+1}] Normalized: {file}")

        # 2. Teknik Computer Vision
        save_image(apply_edge_detection(norm_img), os.path.join(output_folder, f"canny_{file}"))
        save_image(apply_sift_detection(norm_img), os.path.join(output_folder, f"sift_{file}"))

        # 3. Konvolusi
        save_image(apply_gaussian_blur(norm_img), os.path.join(output_folder, f"gaussian_{file}"))
        save_image(apply_sobel_filter(norm_img), os.path.join(output_folder, f"sobel_{file}"))
        save_image(apply_prewitt_filter(norm_img), os.path.join(output_folder, f"prewitt_{file}"))

        # 4. Morfologi
        save_image(apply_opening(norm_img), os.path.join(output_folder, f"opening_{file}"))
        save_image(apply_closing(norm_img), os.path.join(output_folder, f"closing_{file}"))
        save_image(apply_skeletonization(norm_img), os.path.join(output_folder, f"skeleton_{file}"))

        # 5. ORB Keypoints
        _, _, kp_img = detect_keypoints_orb(norm_img)
        save_image(kp_img, os.path.join(output_folder, f"keypoints_{file}"))

        # 6. Matrix 15x15 (gambar pertama saja)
        if index == 0:
            print(f"➡️ Ambil matrix 15x15 dari: {file}")
            matrix = extract_15x15_matrix(norm_img)
            save_15x15_matrix_csv(matrix, os.path.join(output_folder, "matrix15x15.csv"))

            save_image(apply_edge_detection(matrix), os.path.join(output_folder, "canny_matrix15x15.jpg"))
            save_image(apply_sift_detection(matrix), os.path.join(output_folder, "sift_matrix15x15.jpg"))
            save_image(apply_gaussian_blur(matrix), os.path.join(output_folder, "gaussian_matrix15x15.jpg"))
            save_image(apply_sobel_filter(matrix), os.path.join(output_folder, "sobel_matrix15x15.jpg"))
            save_image(apply_prewitt_filter(matrix), os.path.join(output_folder, "prewitt_matrix15x15.jpg"))
            save_image(apply_opening(matrix), os.path.join(output_folder, "opening_matrix15x15.jpg"))
            save_image(apply_closing(matrix), os.path.join(output_folder, "closing_matrix15x15.jpg"))
            save_image(apply_skeletonization(matrix), os.path.join(output_folder, "skeleton_matrix15x15.jpg"))
            _, _, kp_matrix = detect_keypoints_orb(matrix)
            save_image(kp_matrix, os.path.join(output_folder, "keypoints_matrix15x15.jpg"))

    # 7. Pencocokan fitur ORB antara 2 gambar acak
    random_files = random.sample(files, 2)
    print(f"🔀 Matching: {random_files[0]} vs {random_files[1]}")

    img1 = read_image(os.path.join(output_folder, f"normalized_{random_files[0]}"))
    img2 = read_image(os.path.join(output_folder, f"normalized_{random_files[1]}"))

    kp1, desc1, _ = detect_keypoints_orb(img1)
    kp2, desc2, _ = detect_keypoints_orb(img2)

    if desc1 is None or desc2 is None:
        print("Tidak ada deskriptor. Matching gagal.")
        return

    matches = match_features_orb(desc1, desc2)
    matched_img = draw_matches(img1, kp1, img2, kp2, matches)

    match_filename = f"matched_{random_files[0][:-4]}_vs_{random_files[1][:-4]}.jpg"
    save_image(matched_img, os.path.join(output_folder, match_filename))
    print(f"Matching disimpan: {match_filename}")

if __name__ == "__main__":
    main()
