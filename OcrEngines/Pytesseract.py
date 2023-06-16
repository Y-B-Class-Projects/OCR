from PIL import Image
import pytesseract
from OcrEngines.OCR_engine import OCREngine


class Pytesseract(OCREngine):
    def __init__(self, ):
        print('[LOG] [Info] Pytesseract init')
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def get_data(self, image_path):
        pass

    def get_text(self, image_path):
        data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
        text = data['text']
        return text

    def get_boxes(self, image_path: str):
        pass

    def get_text_with_prob(self, image):
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        conf = data['conf']
        text = data['text']
        return [(text[i], conf[i] / 100) for i in range(len(conf)) if conf[i] > 0]
