#这里先记一下思路

'''
Target: 统计图片中直线笔画和曲线笔画的个数

1. 二值化图片

# Try1：直线 直线拟合 https://mp.weixin.qq.com/s/8MkOfaOyjEoMBmV51I1mDg
#           计算主方向
Try2：曲线 利用x y坐标双向投影 来区分直线和曲线
            开闭操作的应用
Try3: 图像处理步骤
        1。
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Pic_data/BW2.jpg',0)
#线检测和弧线检测

import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.二值化图片

img = cv2.imread('./Pic_data/BW2.jpg',0)
    # GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #本来就是gray图片
img_m = cv2.medianBlur(img,5)

AdpThreG = cv2.adaptiveThreshold(img_m,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
# cv2.imshow("AapThreG.jpg", AdpThreG)
# #等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#2.膨胀与腐蚀操作（消除断点）

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(AdpThreG, cv2.MORPH_OPEN, kernel)
result = cv2.resize(opening, (opening.shape[1] // 2, opening.shape[0] // 2))

# cv2.imshow("erosion.jpg",result)
# # cv2.imwrite("erosion.jpg",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#3.二值化
GrayImage = result
ret,thresh1=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY_INV)

# cv2.imshow("1",thresh1)
cv2.imwrite("threhold_1.jpg",thresh1)
# cv2.imshow("2",thresh2)
cv2.imwrite("threhold2.jpg",thresh2)
cv2.waitKey()

