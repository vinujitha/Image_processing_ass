import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
im1 = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\shells.tif", cv.IMREAD_GRAYSCALE)
assert im1 is not None
print(im1.size)# Show the histogram
im3 = cv.equalizeHist(im1)
h = np.zeros(256, dtype= np.uint16)
for r in range(im1.shape[0]):
  for c in range(im1.shape[1]):
    h[im1[r,c]]+=1
print(h)
print(h.size)
plt.bar(range(256),h)
plt.show
# Equalize the histogram
g = np.zeros(256, dtype=np.uint16)  # Initialize an array 'g' to store the equalization transformation
total_pixels = im1.size  # Total number of pixels in the image

for m in range(h.shape[0]):  # Iterate over the histogram 'h'
    sigma_sum = np.sum(h[:m+1])  # Calculate the sum of histogram values up to the m-th element
    g[m] = ((h.size-1) * sigma_sum) // total_pixels  # Calculate the equalization transformation value for the m-th intensity

print(g)
im2 = cv.LUT(im1,g.astype(np.uint8))  # Apply the equalization transformation to the image
# Display the original and equalized images
cv.imshow("Original Image", im1)
cv.imshow("Equalized Image", im2)
cv.imshow("inbuild", im3)
cv.waitKey(0)
cv.destroyAllWindows()

#Plot histograms
plt.hist(im1.ravel(), 256, [0, 256], color='r', alpha=0.5, label='Original Image')
plt.hist(im2.ravel(), 256, [0, 256], color='b', alpha=0.5, label='Equalized Image')
plt.legend(loc='upper right')
plt.show()
