import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = "E:/5th semester/Image Processing/assignment_01_images/assignment_01_images/daisy.jpg"
image = cv2.imread(image_path)

# Create a mask of zeros with dimensions same as the image
mask = np.zeros(image.shape[:2], np.uint8)

# Initialize background and foreground models
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# Define rectangle for initial segmentation
rect = (50,50,900,500)

# Apply GrabCut algorithm for the image
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to identify sure foreground and possible background
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# Apply the mask to the original image
segmented_image = image * mask2[:,:,np.newaxis]

# Create the foreground and background images
foreground_image = np.where(mask2[:,:,np.newaxis] == 1, image, 255)
background_image = np.where(mask2[:,:,np.newaxis] == 0, image, 0)

# Display the segmentation mask, foreground image, and background image
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Segmentation Mask')
plt.imshow(mask2, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Foreground Image')
plt.imshow(cv2.cvtColor(foreground_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Background Image')
plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
