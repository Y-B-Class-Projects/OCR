from easyocr import easyocr
from OCR_engine import OCREngine


class EasyOCR(OCREngine):
    def __init__(self, lang_list, rec_network=None):
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

