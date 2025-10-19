import cv2
import matplotlib.pyplot as plt

def feature_matching(img1_path, img2_path, method="SIFT"):
    # Load images in grayscale
    image1 = cv2.imread(img1_path)
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    if img1 is None or img2 is None:
        raise ValueError("Image path invalid or image not found.")

    # Choose feature detector
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=1000)
    else:
        raise ValueError("Choose method as 'SIFT' or 'ORB'")

    # Detect keypoints and compute descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Match descriptors
    if method == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:  # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12,6))
    plt.imshow(result)
    plt.title(f"{method} Feature Matching")
    plt.axis('off')
    plt.show()

    return matches, kp1, kp2

# Example usage
matches, kp1, kp2 = feature_matching("cola.png", "temp.png", method="SIFT")
matches, kp1, kp2 = feature_matching("cola.png", "temp.png", method="ORB")
