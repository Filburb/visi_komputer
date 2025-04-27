import cv2

def detect_keypoints_orb(image):
    """Deteksi keypoints menggunakan ORB"""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)
    return keypoints, descriptors, img_with_keypoints

def match_features_orb(desc1, desc2):
    """Pencocokan fitur menggunakan Brute Force Matcher"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def draw_matches(image1, kp1, image2, kp2, matches, max_matches=30):
    """Menggambar hasil pencocokan fitur"""
    matched_img = cv2.drawMatches(image1, kp1, image2, kp2, matches[:max_matches], None, flags=2)
    return matched_img
