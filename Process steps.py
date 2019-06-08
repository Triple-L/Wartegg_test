#Process the scanned images


import os
import cv2
import numpy as np
from PIL import Image
from my_utils import *
'''
1.Find out the boarder of the WT samples
    Big black boarders. Cut it into 8 pictures for following steps.
'''

pic_dir = './images_searched/'

im_input = load_img(pic_dir)
im4=im_input[4]
cv2.imshow("im.jpg",im_input[4])
cv2.waitKey(0)
cv2.destroyAllWindows()

# #Try corner detection
# im4 = im4.astype('uint8')
# gray = cv2.cvtColor(im4,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
#
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# im4[dst>0.01*dst.max()]=[0,0,255]
#
# cv2.imshow('dst',im4)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
