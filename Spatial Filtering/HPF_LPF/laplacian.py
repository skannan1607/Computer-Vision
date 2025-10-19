# HIGH PASS FILTER

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your automotive part image

img = cv2.imread("screw.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur (to remove noise)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Laplacian filter (to detect surface scratches/dents)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
laplacian = cv2.convertScaleAbs(laplacian)



# High-pass filtering (to enhance fine details)
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
highpass = cv2.filter2D(gray, -1, kernel)

# Show results
titles = ["Original", "Gray", "Blurred", "Laplacian", "High-pass"]
images = [img, gray, blurred, laplacian, highpass]

plt.figure(figsize=(12,8))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")
plt.show()
