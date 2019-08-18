import cv2
from PIL import Image
img = cv2.imread("./pic_data/BW_black_1.jpg",0)
# cv2.line(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1, 1)
import numpy as np
import matplotlib
import os

from PIL import Image
import sys


def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    count = 0
    for j in range(0, 3):
        for i in range(0, 3):
            count += 1
            box = (i * item_width, j * item_width, (i + 1) * item_width, (j + 1) * item_width)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def save_images(image_list):
    index = 1
    for image in image_list:
        image.save("./save_path/"+str(index) + '.jpg')
        index += 1


if __name__ == '__main__':
    file_path = "001.jpg"
    # 打开图像
    image = Image.open("./pic_data/BW_black_1.jpg")
    # 将图像转为正方形，不够的地方补充为白色底色
    # image=fill_image(image)
    # 分为图像
    image_list = cut_image(image)
    # 保存图像
    save_images(image_list)
