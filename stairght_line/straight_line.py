import cv2
import numpy as np

gray_img = cv2.imread("./BW2.jpg",0)

img_m = cv2.medianBlur(gray_img,5)

AdpThreG = cv2.adaptiveThreshold(img_m,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,9,2)

kernel = np.ones((3,3),np.uint8)
erd_img=cv2.erode(AdpThreG,kernel,iterations = 1)

AT_R = cv2.adaptiveThreshold(erd_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,2)

cv2.imshow("A",AT_R)
cv2.waitKey()



