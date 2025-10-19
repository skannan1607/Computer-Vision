
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("hole.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_eq = clahe.apply(blur)

# Otsu's Thresholding
_, otsu_thresh = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# Adaptive Thresholding
adaptive_mean = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=25, C=10)

adaptive_gaussian = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=10)

# Show results
titles = ['Original Image', 'Otsu Threshold', 'Adaptive Threshold', 'Adaptive Mean', 'Adaptive Gaussian']
images = [gray, otsu_thresh, adaptive_thresh, adaptive_mean, adaptive_gaussian]

plt.figure(figsize=(12,5))
for i in range(5):
    plt.subplot(1,5,i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.axis('off')
plt.show()
