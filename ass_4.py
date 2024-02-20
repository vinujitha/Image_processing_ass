import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
im1 = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\shells.tif", cv.IMREAD_GRAYSCALE)
assert im1 is not None
print(im1.size)# Show the histogram
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
    sigma_sum = np.sum(h[:m+1])  
    g[m] = ((h.size-1) * sigma_sum) // total_pixels  
#print(g)
im2 = cv.LUT(im1,g.astype(np.uint8))  # Apply the equalization transformation to the image
# Display the original and equalized images
im3 = cv.equalizeHist(im1)

#Plot histograms
plt.hist(im1.ravel(), 256, [0, 256], color='r', alpha=0.5, label='Original Image')
plt.hist(im2.ravel(), 256, [0, 256], color='b', alpha=0.5, label='Equalized Image')
plt.legend(loc='upper right')
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\histrograms.png")
plt.show()

# Display the original and equalized images as subplots
plt.figure(figsize=(10, 5))

# Original image subplot
plt.subplot(1, 3, 1)
plt.imshow(im1, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Equalized image using custom function subplot
plt.subplot(1, 3, 2)
plt.imshow(im2, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Equalized image using OpenCV built-in function subplot
plt.subplot(1, 3, 3)
plt.imshow(im3, cmap='gray')
plt.title('OpenCV Equalized Image')
plt.axis('off')

plt.tight_layout()
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\output_images.png")
plt.show()