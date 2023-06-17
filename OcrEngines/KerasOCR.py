import os
import keras_ocr
from OcrEngines.OCR_engine import OCREngine
import cv2


class KerasOCR(OCREngine):
    def __init__(self, ):
        self.pipeline = keras_ocr.pipeline.Pipeline()

    def get_data(self, image_path):
        pass

    def get_text(self, image):
        if image.ndim == 2:     # gray image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # keras_ocr cannot take gray image!
        return [i[0].upper() for i in self.pipeline.recognize([image])[0]]

    def get_boxes(self, image_path: str):
        pass

    def get_text_with_prob(self, image):
        text = self.get_text(image)
        return [(text[i], 1) for i in range(len(text))]


if __name__ == '__main__':
    k = KerasOCR()
    image_path = "../data/26C-085.24.jpg"
    image = cv2.imread(image_path)
    print(k.get_text_with_prob(image))
