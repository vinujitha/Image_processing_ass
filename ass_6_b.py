import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\einstein.png", cv.IMREAD_GRAYSCALE)
assert im1 is not None

# Sobel kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

hw = sobel_x.shape[0] // 2

rows, cols = im1.shape[0], im1.shape[1]
im_sobel_x = np.zeros(im1.shape, dtype=np.float32)
im_sobel_y = np.zeros(im1.shape, dtype=np.float32)

for i in range(hw, rows - hw):
    for j in range(hw, cols - hw):
        # Applying Sobel filter for x direction
        im_sobel_x[i, j] = np.dot(im1[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten(), sobel_x.flatten())
        # Applying Sobel filter for y direction
        im_sobel_y[i, j] = np.dot(im1[i - hw:i + hw + 1, j - hw:j + hw + 1].flatten(), sobel_y.flatten())


fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(im1, vmin=0, vmax=255, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(im_sobel_x + 128, vmin=0, vmax=255, cmap='gray')
ax[1].set_title('Sobel X')
ax[2].imshow(im_sobel_y + 128, vmin=0, vmax=255, cmap='gray')
ax[2].set_title('Sobel Y')

plt.show()