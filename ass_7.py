#part a
import cv2 as cv
import matplotlib.pyplot as plt
# Read the image
image = cv.imread("C:\\Users\ASUS\\Desktop\\rice_gaussian_noise.png")

# Apply Gaussian blur to remove noise
denoised_image = cv.GaussianBlur(image, (5, 5), 0)

# Display the original and denoised images
fig1, axes = plt.subplots(1,2, sharex='all',sharey='all',figsize=(18,18))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(denoised_image)
axes[1].set_title('Filtered Image')
plt.show()

#part b
# Read the image with salt-and-pepper noise
img_2 = cv.imread("C:\\Users\\ASUS\\Desktop\\rice_salt_pepper_noise.png", 0)  # Read as grayscale

# Apply median blur to remove salt-and-pepper noise
denoised_img_2 = cv.medianBlur(image, 5)  # Kernel size is 5x5, adjust as needed

# Display the original and denoised images
fig2, axes = plt.subplots(1,2, sharex='all',sharey='all',figsize=(18,18))
axes[0].imshow(img_2,cmap='gray')
axes[0].set_title('Original Image')
axes[0].set_xticks([]), axes[0].set_yticks([])
axes[1].imshow(denoised_img_2)
axes[1].set_title('Filtered Image')
plt.show()
