import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
im = cv.imread("C:\\Users\\ASUS\\Desktop\\einstein.png", cv.IMREAD_REDUCED_GRAYSCALE_2)
assert im is not None, "Image not found"
array1 = np.array([[1],[2], [1]], dtype='float')  # Shape: (3,1)
array2 = np.array([[1, 0, -1]], dtype='float')  # Shape: (1,3)
# Perform the multiplication
kernel = np.matmul(array1, array2)
img_1 = cv.filter2D(im, -1, kernel)
#print(kernel)
fig, axes = plt.subplots(1,2, sharex='all',sharey='all',figsize=(18,18))
axes[0].imshow(im, cmap ='gray')
axes[0].set_title('Original Image')
axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(img_1, cmap ='gray', vmin=0, vmax=100)
axes[1].set_title('Filtered Image')
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\sobel_own.png")
plt.show()
