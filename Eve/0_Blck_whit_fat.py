import cv2
import numpy as np
#read grey img
img = cv2.imread("./pic_data/BW_black_2.jpg",0)
#adptive threhold Gaussian
dst = cv2.medianBlur(img,7)
th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,2)

kernel = np.ones((3, 3), np.uint8)

img_erosion = cv2.erode(th3, kernel, iterations=2)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)

imagem = cv2.bitwise_not(img_dilation)
dst = cv2.medianBlur(imagem,3)
cv2.imshow("adp",dst)
cv2.imwrite("result_bwf.jpg",dst)
cv2.waitKey()


