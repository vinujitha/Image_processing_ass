import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\highlights_and_shadows.jpg", cv.IMREAD_COLOR)

# Convert the image from BGR to LAB color space
lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
#  Getting L, a, and b channels by using split
L, a, b = cv.split(lab_img)
# Apply gamma correction to the L channel (L plane)
gamma_value = 2.1 # stated gamma value
L_corrected = np.power(L / 255.0, gamma_value) * 255.0
L_corrected = np.clip(L_corrected, 0, 255).astype(np.uint8)
# combained the corrected L channel with the original a and b channels
lab_corrected_image = cv.merge((L_corrected, a, b))
# Convert the image back to the BGR color space
corrected_img = cv.cvtColor(lab_corrected_image, cv.COLOR_LAB2RGB)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')
# Plot the transformed image
axes[1].imshow(corrected_img)
axes[1].set_title('Gamma corrected Image')
axes[1].axis('off')
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\gamma_correction.png")
plt.show()

# Create a separate figure for histograms
plt.figure(figsize=(10, 5))

# Plot histograms for L, a, and b channels of original image
plt.subplot(1, 2, 1)
plt.hist(L.flatten(), bins=256, range=(0, 255), color='r', alpha=0.5, label='L')
plt.hist(a.flatten(), bins=256, range=(0, 255), color='g', alpha=0.5, label='a')
plt.hist(b.flatten(), bins=256, range=(0, 255), color='b', alpha=0.5, label='b')
plt.title('Original Image Histograms')
plt.legend()
# Plot histograms for L, a, and b channels of corrected image
plt.subplot(1, 2, 2)
plt.hist(L_corrected.flatten(), bins=256, range=(0, 255), color='r', alpha=0.5, label='L')
plt.hist(a.flatten(), bins=256, range=(0, 255), color='g', alpha=0.5, label='a')
plt.hist(b.flatten(), bins=256, range=(0, 255), color='b', alpha=0.5, label='b')
plt.title('Corrected Image Histograms')
plt.legend()

plt.tight_layout()
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\gamma_correction_histograms.png")
plt.show()