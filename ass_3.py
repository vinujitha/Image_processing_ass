import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# Read the input image
img1 = cv.imread("C:\\Users\\ASUS\\Desktop\\spider.png", cv.IMREAD_COLOR)

# Display the input image
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

# Convert image to HSV
hsv_image = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
hue_plane, saturation_plane, value_plane = cv.split(hsv_image)

# Create a figure for images using subplot
plt.figure(figsize=(10, 5))

# Plot the Hue plane
plt.subplot(1, 3, 1)
plt.imshow(hue_plane, cmap='hsv', vmin=0, vmax=179)  # Hue values range from 0 to 179
plt.title('Hue Plane')
plt.axis('off')

# Plot the Saturation plane
plt.subplot(1, 3, 2)
plt.imshow(saturation_plane, cmap='hsv', vmin=0, vmax=255)  # Saturation values range from 0 to 255
plt.title('Saturation Plane')
plt.axis('off')

# Plot the Value plane
plt.subplot(1, 3, 3)
plt.imshow(value_plane, cmap='hsv', vmin=0, vmax=255)  # Value values range from 0 to 255
plt.title('Value Plane')
plt.axis('off')

plt.tight_layout()
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\images_subplot.png")
plt.show()

# Extract saturation channel from the hsv plans
saturation = hsv_image[:,:,1]
# Define parameters
a = 0.6
b = 70
# Apply the function to the saturation plane
new_saturation = np.minimum(saturation + a * 128 * np.exp(-((saturation - 128)**2) / (2 * b**2)), 255).astype(np.uint8)

# Replace the saturation plane in the HSV image
hsv_image[:,:,1] = new_saturation
# Convert back to BGR
im_vibrent = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

# Convert BGR images to RGB for plotting
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
im_vibrent_rgb = cv.cvtColor(im_vibrent, cv.COLOR_BGR2RGB)

# Create a figure for images using subplot
plt.figure(figsize=(10, 5))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(img1_rgb)
plt.title('Original Image')
plt.axis('off')

# Plot the vibrant image
plt.subplot(1, 2, 2)
plt.imshow(im_vibrent_rgb)
plt.title('Vibrant Image')
plt.axis('off')

plt.tight_layout()
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\result_subplot.png")
plt.show()

# Display the result
cv.imshow('Result', im_vibrent)
cv.waitKey(0)
cv.destroyAllWindows()
# Define the intensity transformation function
def intensity_transformation(input_intensity, a, b):
    return np.minimum(input_intensity + a * 128 * np.exp(-((input_intensity - 128)**2) / (2 * b**2)), 255).astype(np.uint8)

# Create input intensities
input_intensities = np.arange(256)

# Apply intensity transformation to input intensities
output_intensities = intensity_transformation(input_intensities, a, b)

# Plot the intensity transformation graph
plt.plot(input_intensities, output_intensities, color='black')
plt.title('Intensity Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.grid(True)
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\intensity_trs.png")
plt.show()
