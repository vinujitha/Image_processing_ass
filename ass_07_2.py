import numpy as np
import cv2 as cv

small_img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\a1q5images\\im01small.png", cv.IMREAD_COLOR)
assert small_img is not None

zero_matrix = np.zeros((2 * small_img.shape[0] - 1, 2 * small_img.shape[1] - 1, 3), dtype=np.uint8)

for i in range(small_img.shape[0]):
    for j in range(small_img.shape[1]):
        zero_matrix[2*i, 2*j] = small_img[i, j]

for a in range(0, zero_matrix.shape[0]):
    for b in range(0, zero_matrix.shape[1] - 1):
        if b % 2 != 0:
            zero_matrix[a, b] = (zero_matrix[a, b-1] + zero_matrix[a, b+1]) // 2

for a in range(0, zero_matrix.shape[0] - 1):
    for b in range(0, zero_matrix.shape[1]):
        if a % 2 != 0:
            zero_matrix[a, b] = (zero_matrix[a-1, b] + zero_matrix[a+1, b]) // 2

#print(zero_matrix)
cv.imshow("small" ,small_img)
cv.imshow("near_neighbor", zero_matrix)
cv.waitKey(0)
cv.destroyAllWindows()            
