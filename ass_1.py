import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(220,170),(220,220),(255,240)])
t1 = np.linspace(0, c[0,1], c[0,0]-0).astype('uint8')
print(len(t1))
t2 = np.linspace(c[1,1], c[2,1], c[2,0]-(c[0,0]-1)).astype('uint8')
print(len(t2))

transform = np.concatenate((t1,t2), axis=0).astype('uint8')
print(len(transform))
#print(transform)

fig, ax =plt.subplots()
ax.plot(transform)
ax.set_xlabel(r'Input, $f(mathbf{x})$')
ax.set_ylabel('Output, $\mathrm{T}[f\mathbf{x})]$')
ax.set_xlim(0,255)
ax.set_ylim(0,240)
#plt.show

img_org = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\margot_golden_gray.jpg", cv.IMREAD_GRAYSCALE)
#cv.imshow("image", img_org)
#cv.waitKey(0)
#cv.destroyAllWindows()
#img = cv.cvtColor(img_org, cv.COLOR_BGR2GRAY)

#fig, ax = plt.subplots()
#ax.imshow(img_org, cmap = 'gray')
#ax.set_title('Image')
#plt.show()

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
plt.show()

