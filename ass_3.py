import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img1 = cv.imread("C:\\Users\\ASUS\\Desktop\\spider.png", cv.IMREAD_COLOR)
cv.namedWindow('Image', cv.WINDOW_NORMAL)
cv.imshow('Image', img1)
cv.waitKey(0)
cv.destroyAllWindows()

# Split the image into its RGB channels
r, g, b = cv.split(img1)

# Initialize histograms for each channel
h_r = np.zeros(256, dtype=np.uint16)
h_g = np.zeros(256, dtype=np.uint16)
h_b = np.zeros(256, dtype=np.uint16)

# Compute histograms for each channel
for channel in [r, g, b]:
    for pixel_value in channel.flatten():
        if channel is r:
            h_r[pixel_value] += 1
        elif channel is g:
            h_g[pixel_value] += 1
        elif channel is b:
            h_b[pixel_value] += 1

# Plot histograms in separate subplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(256), h_r, color='red')
plt.title('Red Channel')

plt.subplot(1, 3, 2)
plt.bar(range(256), h_g, color='green')
plt.title('Green Channel')

plt.subplot(1, 3, 3)
plt.bar(range(256), h_b, color='blue')
plt.title('Blue Channel')

plt.show()

hsv_image = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
hue_plane, saturation_plane, value_plane = cv.split(hsv_image)
cv.imshow('Hue Plane', hue_plane)
cv.imshow('Saturation Plane', saturation_plane)
cv.imshow('Value Plane', value_plane)
cv.waitKey(0)
cv.destroyAllWindows()


# Convert image to HSV
im_hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

# Extract saturation channel
saturation = im_hsv[:,:,1]

# Define parameters
a = 0.6
b = 70

# Apply the function to the saturation plane
new_saturation = np.minimum(saturation + a * 128 * np.exp(-((saturation - 128)**2) / (2 * b**2)), 255).astype(np.uint8)

# Replace the saturation plane in the HSV image
im_hsv[:,:,1] = new_saturation

# Convert back to BGR
im_result = cv.cvtColor(im_hsv, cv.COLOR_HSV2BGR)

# Display the result
cv.imshow('Result', im_result)
cv.waitKey(0)
cv.destroyAllWindows()
