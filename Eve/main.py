import os
import cv2
import numpy as np
img_path = "./pic_data/"
save_path = "./save_path"
if not os.path.exists(img_path):
    os.mkdir(save_path)

file_names = os.listdir(img_path)


# for i in range(len(file_names)):
#     img = cv2.imread(img_path + file_names[i],0)  # 得到文件名
#     #cv2.imwrite(save_path + file_names[i], img)
#     # cv2.imshow("img",img)
#     # cv2.waitKey()
#     h, w = img.shape[:2]
#     print("The img_"+str(i)+"(h,w):",(h,w))
    #cut into 8 pieces


