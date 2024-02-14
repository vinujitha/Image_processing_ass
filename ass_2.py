import cv2 as cv
import numpy as np

gamma = 1.4
f = cv.imread(r"E:\5th semester\Image Processing\assignment_01_images\assignment_01_images\highlights_and_shadows.jpg", cv.IMREAD_COLOR)
assert f is not None

cv.imshow("Original", f)

# Calculate gamma correction lookup table
t = np.array([(i/255.0)**(1/gamma)*255 for i in np.arange(0, 256)]).astype(np.uint8)

# Apply gamma correction using lookup table
g = cv.LUT(f, t)

cv.imshow("Gamma_Corrected", g)
cv.waitKey(0)
cv.destroyAllWindows()

