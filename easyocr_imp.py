# import os
#
# import cv2
# import easyocr
# import numpy as np
# from pytesseract import pytesseract, Output
#
# from tools import get_grayscale, thresholding, process_image, remove_red
#
# # reader = easyocr.Reader(['vi', 'en'])  # this needs to run only once to load the model into memory
# reader = easyocr.Reader(['en'], recog_network='best_accuracy')
# image, gray, thresh = process_image('data/LD_06.jpg')
#
# unred_images = remove_red(image)
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # image = cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=0, sigmaY=0)
# # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
# # image = clahe.apply(image)
# _, image = cv2.threshold(image, 200, 255, cv2.THRESH_OTSU)
#
# result = reader.readtext(image)
#
# min_x, min_y, max_x, max_y = image.shape[0], image.shape[1], 0, 0
#
# for res in result:
#     box, text, prob = res
#     all_x = [point[0] for point in box]
#     all_y = [point[1] for point in box]
#     min_x = min(min_x, min(all_x))
#     min_y = min(min_y, min(all_y))
#     max_x = max(max_x, max(all_x))
#     max_y = max(max_y, max(all_y))
#     print(text)
#
#     box = [(int(a), int(b)) for (a, b) in box]
#     cv2.line(image, box[0], box[1], (0, 255, 0), 2)
#     cv2.line(image, box[1], box[2], (0, 255, 0), 2)
#     cv2.line(image, box[2], box[3], (0, 255, 0), 2)
#     cv2.line(image, box[3], box[0], (0, 255, 0), 2)
#
# min_x = int(min_x) - 2
# min_y = int(min_y) - 2
# max_x = int(max_x) + 2
# max_y = int(max_y) + 2
#
# min_x = max(min_x, 0)
# min_y = max(min_y, 0)
# max_x = min(max_x, image.shape[1])
# max_y = min(max_y, image.shape[0])
#
# image_with = 1000
# image_height = int(image_with * (max_y - min_y) / (max_x - min_x))
#
# pts1 = np.float32([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
# pts2 = np.float32([[0, 0], [0, image_height], [image_with, 0], [image_with, image_height]])
#
# # Apply Perspective Transform Algorithm
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# image_new = cv2.warpPerspective(image, matrix, (image_with, image_height))
#
# cv2.line(image, (min_x, min_y), (min_x, max_y), (0, 0, 255), 2)
# cv2.line(image, (min_x, min_y), (max_x, min_y), (0, 0, 255), 2)
# cv2.line(image, (max_x, min_y), (max_x, max_y), (0, 0, 255), 2)
# cv2.line(image, (min_x, max_y), (max_x, max_y), (0, 0, 255), 2)
#
# cv2.imshow('image', image)
# cv2.imshow('image_new', image_new)
# cv2.waitKey(0)
#
from easyocr import easyocr
from OCR_engine import OCREngine


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class EasyOCR(metaclass=Singleton):
    def __init__(self, lang_list, rec_network=None):
        print('EasyOCR init', lang_list, rec_network)
        if rec_network is None:
            self.reader = easyocr.Reader(lang_list)
        else:
            self.reader = easyocr.Reader(lang_list, recog_network=rec_network)

    def get_data(self, image_path):
        return self.reader.readtext(image_path)

    def get_text(self, image_path):
        result = self.reader.readtext(image_path)
        return [res[1] for res in result]

    def get_boxes(self, image_path: str):
        result = self.reader.readtext(image_path)
        return [res[0] for res in result]

    def get_text_with_prob(self, image_path):
        result = self.reader.readtext(image_path)
        return [(res[1], res[2]) for res in result]
