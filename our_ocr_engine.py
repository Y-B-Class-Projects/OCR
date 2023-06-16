from easyocr_imp import EasyOCR


class OurOcrEngine(EasyOCR):
    def __init__(self, lang_list):
        super().__init__(lang_list, 'best_accuracy')
        print('EasyOCR init', lang_list)
