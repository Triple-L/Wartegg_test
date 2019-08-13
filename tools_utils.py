#线检测和弧线检测

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Pic_data/BW2.jpg',0)
img = cv2.medianBlur(img,5)

AdpThreG = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

print(AdpThreG)
# cv2.imshow("AapThreG.jpg", AdpThreG)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()



