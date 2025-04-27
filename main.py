import os
import cv2
from detection.image_processing import (
    read_image,
    extract_15x15_matrix,
    save_image,
    save_15x15_matrix_csv
)
from detection.image_preprocessing import enhance_contrast
from detection.teknik import (
    apply_edge_detection,
    apply_harris_corner_detection
)
from detection.konvolusi import (
    apply_gaussian_blur,
    apply_sobel_filter,
    apply_prewitt_filter
)
from detection.morfologi import (
    apply_dilation,
    apply_erosion,
    apply_skeletonization
)
from detection.feature import (
    detect_keypoints_orb,
    match_features_orb,
    draw_matches
)

def ensure_folder(folder_path):
    """Buat folder jika belum ada"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def main():
    input_folder = "Hugh Jackman"    # Folder input gambar
    output_folder = "output"          # Folder output hasil
    ensure_folder(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files or len(files) < 2:
        print("Minimal 2 gambar diperlukan di folder input untuk matching!")
        return

    for index, file in enumerate(files):
        img_path = os.path.join(input_folder, file)
        img = read_image(img_path)

        # 1. Peningkatan kontras
        enhanced_img = enhance_contrast(img, alpha=1.5)

        # 2. Simpan gambar hasil enhancement
        enhanced_save_path = os.path.join(output_folder, f"enhanced_{file}")
        save_image(enhanced_img, enhanced_save_path)
        print(f"Disimpan hasil peningkatan kontras: {enhanced_save_path}")

        # 3. Vision Processing (Edge Detection + Harris)
        edges = apply_edge_detection(enhanced_img)
        save_image(edges, os.path.join(output_folder, f"edge_enhanced_{file}"))
        
        harris = apply_harris_corner_detection(enhanced_img)
        save_image(harris, os.path.join(output_folder, f"harris_enhanced_{file}"))

        # 4. Convolution Processing (Gaussian, Sobel, Prewitt)
        gaussian = apply_gaussian_blur(enhanced_img)
        save_image(gaussian, os.path.join(output_folder, f"gaussian_enhanced_{file}"))

        sobel = apply_sobel_filter(enhanced_img)
        save_image(sobel, os.path.join(output_folder, f"sobel_enhanced_{file}"))

        prewitt = apply_prewitt_filter(enhanced_img)
        save_image(prewitt, os.path.join(output_folder, f"prewitt_enhanced_{file}"))

        # 5. Morfologi Processing (Dilation, Erosion, Skeletonization)
        dilated = apply_dilation(enhanced_img)
        save_image(dilated, os.path.join(output_folder, f"dilated_enhanced_{file}"))

        eroded = apply_erosion(enhanced_img)
        save_image(eroded, os.path.join(output_folder, f"eroded_enhanced_{file}"))

        skeleton = apply_skeletonization(enhanced_img)
        save_image(skeleton, os.path.join(output_folder, f"skeleton_enhanced_{file}"))

        # 6. Fitur Deteksi (ORB Keypoints)
        keypoints, descriptors, img_with_keypoints = detect_keypoints_orb(enhanced_img)
        save_image(img_with_keypoints, os.path.join(output_folder, f"keypoints_enhanced_{file}"))

        # 7. Matrix 15x15 dari gambar pertama
        if index == 0:
            print(f"➡️ Matrix 15x15 diambil dari file: {file}")

            matrix_15x15 = extract_15x15_matrix(enhanced_img)
            matrix_save_path = os.path.join(output_folder, "matrix15x15.csv")
            save_15x15_matrix_csv(matrix_15x15, matrix_save_path)
            print(f"Matrix 15x15 disimpan di: {matrix_save_path}")

            # Vision processing di matrix 15x15
            edges_matrix = apply_edge_detection(matrix_15x15)
            save_image(edges_matrix, os.path.join(output_folder, "edge_matrix15x15.jpg"))

            harris_matrix = apply_harris_corner_detection(matrix_15x15)
            save_image(harris_matrix, os.path.join(output_folder, "harris_matrix15x15.jpg"))

            # Convolution processing di matrix 15x15
            gaussian_matrix = apply_gaussian_blur(matrix_15x15)
            save_image(gaussian_matrix, os.path.join(output_folder, "gaussian_matrix15x15.jpg"))

            sobel_matrix = apply_sobel_filter(matrix_15x15)
            save_image(sobel_matrix, os.path.join(output_folder, "sobel_matrix15x15.jpg"))

            prewitt_matrix = apply_prewitt_filter(matrix_15x15)
            save_image(prewitt_matrix, os.path.join(output_folder, "prewitt_matrix15x15.jpg"))

            # Morfologi processing di matrix 15x15
            dilated_matrix = apply_dilation(matrix_15x15)
            save_image(dilated_matrix, os.path.join(output_folder, "dilated_matrix15x15.jpg"))

            eroded_matrix = apply_erosion(matrix_15x15)
            save_image(eroded_matrix, os.path.join(output_folder, "eroded_matrix15x15.jpg"))

            skeleton_matrix = apply_skeletonization(matrix_15x15)
            save_image(skeleton_matrix, os.path.join(output_folder, "skeleton_matrix15x15.jpg"))

            # Fitur Deteksi di matrix 15x15
            kp_matrix, desc_matrix, img_kp_matrix = detect_keypoints_orb(matrix_15x15)
            save_image(img_kp_matrix, os.path.join(output_folder, "keypoints_matrix15x15.jpg"))

    # 8. Matching fitur antara gambar 1 dan gambar 2
    img1 = read_image(os.path.join(output_folder, f"enhanced_{files[0]}"))
    img2 = read_image(os.path.join(output_folder, f"enhanced_{files[1]}"))

    kp1, desc1, _ = detect_keypoints_orb(img1)
    kp2, desc2, _ = detect_keypoints_orb(img2)

    matches = match_features_orb(desc1, desc2)
    matched_img = draw_matches(img1, kp1, img2, kp2, matches)

    matched_save_path = os.path.join(output_folder, f"matched_{files[0][:-4]}_vs_{files[1][:-4]}.jpg")
    save_image(matched_img, matched_save_path)
    print(f"Disimpan hasil Matching ORB: {matched_save_path}")

if __name__ == "__main__":
    main()
