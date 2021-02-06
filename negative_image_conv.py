import os
from cv2 import cv2 as cv
from car_detection import IMG

train_path = "./car_detection/train_img/no_cars"
test_path = "./car_detection/test_img/no_cars"
home = os.getenv('HOME')
src_path = home + "/Downloads/no_cars"

src_folders = os.listdir(src_path)
for src_folder in src_folders:
    img_path = os.path.join(src_path, src_folder)
    src_images = os.listdir(img_path)
    print(img_path)
    train_images = src_images[50:]
    test_images = src_images[:50]

    for im in train_images:
        img = os.path.abspath(os.path.join(img_path, im))
        tgt = os.path.abspath(os.path.join(
            train_path, src_folder.lower() + '_' + im))
        print('Writing Image {} to {}'.format(img, tgt))
        cv_im = IMG(img)
        cv_im.convert_for_tf(tgt)

    for im in test_images:
        img = os.path.abspath(os.path.join(img_path, im))
        tgt = os.path.abspath(os.path.join(
            test_path, src_folder.lower() + '_' + im))
        print('Writing Image {} to {}'.format(img, tgt))
        cv_im = IMG(img)
        cv_im.convert_for_tf(tgt)
