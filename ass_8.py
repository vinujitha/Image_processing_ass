import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv.imread("E://5th semester//Image Processing//assignment_01_images//assignment_01_images//daisy.jpg")
# Create a mask of zeros with dimensions same as the image
mask = np.zeros(image.shape[:2], np.uint8)
# Initialize background and foreground models
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
# Define rectangle for initial segmentation
rect = (50, 50, 900, 500)
# Apply GrabCut algorithm for the image
cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
# Modify the mask to identify sure foreground and possible background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# Create the foreground and background images
foreground_image = np.where(mask2[:, :, np.newaxis] == 1, image, 0)
background_image = np.where(mask2[:, :, np.newaxis] == 0, image, 0)
burred_background = cv.blur(background_image, (20,20))
# Create a black background image with the same dimensions as the original image
black_background = np.zeros_like(image)
# Place the foreground image on the black background
composite_image1 = cv.add(black_background, foreground_image)
# Add the original foreground and background images using cv2.add
composite_image = cv.add(composite_image1, burred_background)

# Display the segmentation mask, foreground image, background image, and composite image
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title('Segmentation Mask')
plt.imshow(mask2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Foreground Image')
plt.imshow(cv.cvtColor(foreground_image, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Background Image')
plt.imshow(cv.cvtColor(background_image, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Burred Background Image')
plt.imshow(cv.cvtColor(burred_background, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Orginal Image')
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Composite Image')
plt.imshow(cv.cvtColor(composite_image, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\outputs.png")
plt.show()
