# LOW PASS FILTER
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("screw.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur (to remove noise)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

cv2.imwrite("gaussian_filtered.jpg", blurred)
plt.imshow(blurred, cmap='gray')
plt.axis('off')
plt.show()
