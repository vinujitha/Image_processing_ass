import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im = cv.imread("C:\\Users\\ASUS\\Desktop\\einstein.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
assert im is not None, "Image not found"

# Apply Sobel filter in the horizontal direction
sobel_ho1 = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3)  
sobel_ho2 = np.absolute(sobel_ho1)
sobel_horizontal = np.uint8(255 * sobel_ho2 / np.max(sobel_ho2))

# Apply Sobel filter in the vertical direction
sobel_ver1 = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)  
sobel_ver2 = np.absolute(sobel_ver1)
sobel_vertical = np.uint8(255 * sobel_ver2 / np.max(sobel_ver2))

# Plot the original and sobel filted images
fig, axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(18, 18))
axes[0].imshow(im, cmap='gray')
axes[0].set_title('Original Image')
axes[0].set_xticks([]), axes[0].set_yticks([])

axes[1].imshow(sobel_horizontal, cmap='gray')
axes[1].set_title('Horizontal Sobel Filter image')
axes[1].set_xticks([]), axes[1].set_yticks([])

axes[2].imshow(sobel_vertical, cmap='gray')
axes[2].set_title('Vertical Sobel Filterimage')
axes[2].set_xticks([]), axes[2].set_yticks([])
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\sobel.png")
plt.show()
