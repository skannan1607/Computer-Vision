
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils                      # autocanny from cv2

img = cv2.imread("bridge_concrete.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  Noise reduction (important for cracks)
#  Use (5, 5) for noisy, high-resolution images.
#  Always use an odd kernel size for Gaussian blurs so the central pixel is well-defined.

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny Edge Detector (strong crack localization) use two threshold
canny = cv2.Canny(blurred, threshold1=50, threshold2=150)

#  Use (3, 3) for smaller, less noisy images or where edge sharpness must be retained.
blurred1 = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred1, 10, 200)
tight = cv2.Canny(blurred1, 225, 250)
auto = imutils.auto_canny(blurred1)

#  Display all results
titles = ["Original", "Blurred", "Canny", "Wide", "Tight", "Auto"]
images = [img, blurred, canny, wide, tight, auto]

plt.figure(figsize=(24,12))
for i in range(len(images)):
    plt.subplot(2,4,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")
plt.show()

cv2.waitKey(0)
