import cv2 as cv
import numpy as np

# Load the small image
small_img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\a1q5images\\im04small.jpg", cv.IMREAD_COLOR)
orginal_img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\a1q5images\\im04.jpg", cv.IMREAD_COLOR)
assert small_img is not None, "Image not found"

print(small_img.shape)
print(orginal_img.shape)

scale_factor = 5

# Create a zero matrix with the same size of orginal image
upscaled_height = small_img.shape[0] * scale_factor
upscaled_width = small_img.shape[1] * scale_factor
upscaled_img = np.zeros((upscaled_height, upscaled_width, 3), dtype=np.uint8)
# Algoridam of nearest neighbor interpolation
for y in range(small_img.shape[0]):
    for x in range(small_img.shape[1]):
        upscaled_img[y*scale_factor:(y+1)*scale_factor, x*scale_factor:(x+1)*scale_factor] = small_img[y, x]
print(upscaled_img.shape)
# Display the small image and the upscaled image
cv.imshow("Small Image1", small_img)
cv.imshow("Small Image2", small_img)
cv.imshow("Near Neighbor Interpolated Image", upscaled_img)
cv.waitKey(0)
cv.destroyAllWindows()
# Compute the Sum of Squared Differences (SSD) between the original and scaled images
ssd = np.sum((orginal_img.astype("float") - upscaled_img.astype("float")) ** 2)

# Normalize the SSD
num_pixels = orginal_img.shape[0] * orginal_img.shape[1] * orginal_img.shape[2]
normalized_ssd = ssd / num_pixels

print("Normalized SSD:", normalized_ssd)
