import os
from cv2 import cv2 as cv
from car_detection import IMG

train_path = "./car_detection/train_img/cars"
test_path = "./car_detection/test_img/cars"
home = os.getenv('HOME')
src_path = home + "/Downloads/cars_train"

src_images = os.listdir(src_path)
train_images = src_images[:7001]
test_images = src_images[7001:8001]

for im in train_images:
    img = os.path.abspath(os.path.join(src_path, im))
    tgt = os.path.abspath(os.path.join(train_path, im))
    print('Writing Image {} to {}'.format(img, tgt))
    cv_im = IMG(img)
    cv_im.convert_for_tf(tgt)

for im in test_images:
    img = os.path.abspath(os.path.join(src_path, im))
    tgt = os.path.abspath(os.path.join(test_path, im))
    print('Writing Image {} to {}'.format(img, tgt))
    cv_im = IMG(img)
    cv_im.convert_for_tf(tgt)
