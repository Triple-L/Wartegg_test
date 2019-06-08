

import os
import cv2
from PIL import Image

# Load images
def check_file(data_dir):
    if os.path.exists(data_dir):
            return True
    else:
        print("No such file,please check the dir!")
        return False

def load_img(file_dir):
    img_lst=[]
    for file in os.listdir(file_dir):
        file_name = file_dir+file
        img = cv2.imread(file_name,0)
        img_lst.append(img)
        # print(img_lst)
    return img_lst


