import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Function to apply Otsu's method for image segmentation
def apply_otsu_segmentation(image):
    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply Otsu's method to find the optimal threshold
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Invert the thresholded image (optional depending on application)
    segmented_image = cv.bitwise_not(thresh)
    return segmented_image

# Function to apply morphological operations
def apply_morphological_operations(image):
    # Define kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)
    # Perform closing to fill small holes
    closed_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    # Perform opening to remove small objects
    opened_image = cv.morphologyEx(closed_image, cv.MORPH_OPEN, kernel)
    return opened_image

# Function for the count number of rice grains using connected components
def count_rice_grains(segmented_image):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(segmented_image, connectivity=4)
    num_rice_grains = num_labels - 1
    return num_rice_grains

# Read the images
image_gaussian_noise = cv.imread("C:\\Users\ASUS\\Desktop\\rice_gaussian_noise.png")
image_salt_pepper_noise = cv.imread("C:\\Users\\ASUS\\Desktop\\rice_salt_pepper_noise.png")

# Apply Gaussian blur to remove noise from the image with Gaussian noise
denoised_image_gaussian = cv.GaussianBlur(image_gaussian_noise, (5, 5), 0)

# Apply median blur to remove noise from the image with salt-and-pepper noise
denoised_image_salt_pepper = cv.medianBlur(image_salt_pepper_noise, 5)

# Apply Otsu's method for image segmentation to both denoised images
segmented_image_gaussian = apply_otsu_segmentation(denoised_image_gaussian)
segmented_image_salt_pepper = apply_otsu_segmentation(denoised_image_salt_pepper)

# Apply morphological operations to remove small objects and fill holes in segmented images
processed_segmented_image_gaussian = apply_morphological_operations(segmented_image_gaussian)
processed_segmented_image_salt_pepper = apply_morphological_operations(segmented_image_salt_pepper)

# Invert binary values of segmented images
inverted_segmented_image_gaussian = cv.bitwise_not(processed_segmented_image_gaussian)
inverted_segmented_image_salt_pepper = cv.bitwise_not(processed_segmented_image_salt_pepper)

# Count the number of rice grains using connected components for inverted segmented images
num_rice_grains_gaussian = count_rice_grains(inverted_segmented_image_gaussian)
num_rice_grains_salt_pepper = count_rice_grains(inverted_segmented_image_salt_pepper)

print("Number of rice grains in image with Gaussian noise:", num_rice_grains_gaussian)
print("Number of rice grains in image with salt-and-pepper noise:", num_rice_grains_salt_pepper)

# Display the original, denoised, segmented, and processed segmented images for both images
fig, axes = plt.subplots(2, 2, figsize=(10, 5))

axes[0, 0].imshow(segmented_image_gaussian, cmap='gray')
axes[0, 0].set_title('Segmented Image (Gaussian Noise)')
axes[0, 0].set_xticks([]), axes[0, 0].set_yticks([])
axes[0, 1].imshow(processed_segmented_image_gaussian, cmap='gray')
axes[0, 1].set_title('Processed Segmented Image')
axes[0, 1].set_xticks([]), axes[0, 1].set_yticks([])

axes[1, 0].imshow(segmented_image_salt_pepper, cmap='gray')
axes[1, 0].set_title('Segmented Image (Salt-and-Pepper Noise)')
axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])
axes[1, 1].imshow(processed_segmented_image_salt_pepper, cmap='gray')
axes[1, 1].set_title('Processed Segmented Image')
axes[1, 1].set_xticks([]), axes[1, 1].set_yticks([])

plt.tight_layout()
plt.show()