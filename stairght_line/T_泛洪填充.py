#泛洪填充(二值图像填充)
import cv2 as cv
import numpy as np
def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300] = 255
    cv.imshow("fill_binary", image)
    mask = np.ones([402, 402], np.uint8)   #mask要保证比原图像高和宽都多2
    mask[101:301, 101:301] = 0
    cv.floodFill(image, mask, (200,200), (255 , 0, 0), cv.FLOODFILL_MASK_ONLY) #mask不为0的区域不会被填充，mask为0的区域才会被填充
    cv.imshow("filled_binary", image)
fill_binary()
cv.waitKey(0)
cv.destroyAllWindows()