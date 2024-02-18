import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(220,170),(220,220),(255,240)])
t1 = np.linspace(0, c[0,1], c[0,0]-0).astype('uint8')
print(len(t1))
t2 = np.linspace(c[1,1], c[2,1], c[2,0]-(c[0,0]-1)).astype('uint8')
#print(len(t2))
transform = np.concatenate((t1,t2), axis=0).astype('uint8')
#print(len(transform))

fig, ax =plt.subplots()
ax.plot(transform)
ax.set_xlabel(r'Input, $x$')
ax.set_ylabel('Output, $\mathrm{T}[f\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\intensity_tranformation_table.png")
# Apply intensity tranformation for the image 
img_org = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\margot_golden_gray.jpg", cv.IMREAD_GRAYSCALE)
img_trans = cv.LUT(img_org, transform)
# Plot the original image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_org, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')
# Plot the transformed image
axes[1].imshow(img_trans, cmap='gray')
axes[1].set_title('Transformed Image')
axes[1].axis('off')
plt.savefig("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\New folder\\transformed_image.png")
plt.show()


