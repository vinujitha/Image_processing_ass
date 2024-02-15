import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
small_img = cv.imread("E:\\5th semester\\Image Processing\\assignment_01_images\\assignment_01_images\\a1q5images\\im01small.png", cv.IMREAD_COLOR)
assert small_img is not None
print(small_img.shape)

print(matrix.size)
# going to make a zero matrix of size 6X6
zero_matrix = np.zeros((540, 960,3))
#print(zero_matrix)
for p in range(3):
    for i in range(270):
        a = 2*i
        for j in range(480):
            b = 2*j
            zero_matrix[a,b,p] = small_img[i,j,p]      
            zero_matrix[a, b+1,p] = small_img[i,j,p]
    near_neighbor = zero_matrix
for q in range(3):    
    for i in range(270):
        t = 2*i
        for j in range(960):
            near_neighbor[t+1,j] = zero_matrix[t,j]
print(near_neighbor)
cv.imshow("small" ,small_img)
cv.imshow("near_neighbor", near_neighbor)
cv.waitKey(0)
cv.destroyAllWindows()































