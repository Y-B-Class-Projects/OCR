import os
import keras_ocr
from OcrEngines.OCR_engine import OCREngine


class KerasOCR(OCREngine):
    def __init__(self, ):
        self.pipeline = keras_ocr.pipeline.Pipeline()

    def get_data(self, image_path):
        pass

    def get_text(self, image):
        prediction_groups = self.pipeline.recognize(image)
        return prediction_groups

    def get_boxes(self, image_path: str):
        pass

    def get_text_with_prob(self, image):
        pass


if __name__ == '__main__':
    k = KerasOCR()
    print(k.get_text('./data/12F9-9883.jpg'))
