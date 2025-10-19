
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("apple.png", 0)

# Convert to binary (thresholding)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Structuring element (kernel)
kernel = np.ones((5,5), np.uint8)

# Morphological operations
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
eroded = cv2.erode(binary, kernel, iterations=1)
dilated = cv2.dilate(binary, kernel, iterations=1)

# Plot results
titles = ['Original Grayscale', 'Binary Image', 'Opening', 'Closing', 'Erosion', 'Dilation']
images = [img, binary, opening, closing, eroded, dilated]

plt.figure(figsize=(15, 6))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=12)
    plt.axis('off')

plt.subtitle("Morphological Operations", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
