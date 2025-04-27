import cv2

def enhance_contrast(image, alpha=1.5, beta=0):
    """
    Peningkatan kontras dengan mengalikan pixel dengan alpha dan menambahkan beta.
    Alpha > 1 meningkatkan kontras.
    """
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image
