import numpy as np
import cv2 as cv

def bilinear_interpolation(image, scale_factor):
    # Define the dimensions of the zoomed image
    zoomed_height = int(image.shape[0] * scale_factor)
    zoomed_width = int(image.shape[1] * scale_factor)

    # Create a zero matrix for the zoomed image
    zoomed_img = np.zeros((zoomed_height, zoomed_width, 3), dtype=np.uint8)
    #new_heigh = zoomed_height -1
    # Perform bilinear interpolation
    for i in range(zoomed_height):
        for j in range(zoomed_width):
            # Calculate the corresponding position in the original image
            x = i / scale_factor
            y = j / scale_factor
            x0 = int(x)
            y0 = int(y)
            x1 = min(x0 + 1, image.shape[0] - 1)
            y1 = min(y0 + 1, image.shape[1] - 1)

            # Bilinear interpolation
            dx = x - x0
            dy = y - y0
            zoomed_img[i, j] = (1 - dx) * (1 - dy) * image[x0, y0] + dx * (1 - dy) * image[x1, y0] + \
                               (1 - dx) * dy * image[x0, y1] + dx * dy * image[x1, y1]

    return zoomed_img

# Read the small image
small_img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\a1q5images\\im04small.jpg", cv.IMREAD_COLOR)
orginal_img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\a1q5images\\im04.jpg", cv.IMREAD_COLOR)
#orginal_img = np.zeros((orginal_img.shape[0] + 1, orginal_img.shape[1], 3), dtype=np.uint8)
assert small_img is not None, "Image not found"
print(small_img.shape)
print(orginal_img.shape)
# Define the scaling factor
scale_factor = 5

# Perform bilinear interpolation
zoomed_img = bilinear_interpolation(small_img, scale_factor)

# Display the original and zoomed images
cv.imshow("Small Image", small_img)
cv.imshow("Small Image", small_img)
cv.imshow("Zoomed Image", zoomed_img)
cv.waitKey(0)
cv.destroyAllWindows()

# Compute the Sum of Squared Differences (SSD) between the original and scaled images
ssd = np.sum((orginal_img.astype("float") - zoomed_img.astype("float")) ** 2)

# Normalize the SSD
num_pixels = orginal_img.shape[0] * orginal_img.shape[1] * orginal_img.shape[2]
normalized_ssd = ssd / num_pixels

print("Normalized SSD:", normalized_ssd)
