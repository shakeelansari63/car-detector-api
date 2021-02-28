# Image operations using OpenCV
from cv2 import cv2 as cv


class IMG:
    """Image class for image operations"""

    def __init__(self, img_path):
        self.img = cv.imread(img_path, )

    def to_gray(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def resize(self, width, height):
        self.img = cv.resize(self.img, dsize=(width, height),
                             interpolation=cv.INTER_AREA)

    def get_img(self):
        return self.img

    def save(self, tgt_path):
        cv.imwrite(tgt_path, self.img)

    def convert_for_tf(self, tgt_path):
        self.resize(60, 60)
        self.save(tgt_path)
