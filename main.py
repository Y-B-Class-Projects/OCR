import os
import re
import imutils
import pytesseract
from pytesseract import Output
from tools import *


def main():
    config = '--oem 1 --psm 6'
    plate = r"^.*([0-9][0-9])(([A-Z][A-Z]?-)|(-[A-Z]([0-9]|[A-Z])))(([0-9][0-9][0-9][0-9])|([0-9][0-9][0-9]\.[0-9][0-9]))"
    capacity_format = '^.*(Capacity)|(tích)'
    sit_format = '^.*(Sit)|(Seat)|(ngồi)'

    for im in os.listdir('data'):
        image, gray, thresh = process_image(os.path.join('data', im))
        for img in [image, gray, thresh]:
            d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='vie+eng', config=config)
            for i in range(len(d['text'])):
                if re.match(plate, d['text'][i]) or re.match(capacity_format, d['text'][i]) or re.match(sit_format,
                                                                                                        d['text'][i]):
                    print(d['text'][i], '[', d['conf'][i], ']')
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('img', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # main()

    from easyocr_imp import EasyOCR

    e_ocr = EasyOCR(['en'], rec_network='best_accuracy')
    # text = e_ocr.get_text('data/LD_01.jpg')
    # print(text)

    # DDC-              DDD.DD
    # DDCC-             DDD.DD
    # DD-CD             DDD.DD
    # DD-CC             DDD.DD
    # DDC-                          DDD
    # DDCC-                         DDD
    # DD-CD                                 DDDD
    # DD-CC                                 DDDD
    # DDCD-DDDD
    # DDCD-DDD.DD
    plate = r"(\d{2}[A-Z]-\d{3}\.\d{2})|" \
            r"(\d{2}[A-Z]{2}-\d{3}\.\d{2})|" \
            r"(\d{2}-[A-Z]\d{4}\.\d{2})|" \
            r"(\d{2}-[A-Z]{2}\d{3}\.\d{2})|" \
            r"(\d{2}[A-Z]-\d{3})|" \
            r"(\d{2}[A-Z]{2}-\d{3})|" \
            r"(\d{2}-[A-Z]\d{5})|" \
            r"(\d{2}-[A-Z]{2}\d{4})|" \
            r"(\d{2}[A-Z]\d-\d{4})|" \
            r"(\d{2}[A-Z]\d-\d{3}\.\d{2})"

    for im in os.listdir('trainer/all_data/test'):
        image, gray, thresh = process_image(os.path.join('trainer/all_data/test', im))
        images = [image]
        # images.extend(remove_red(image))
        texts = []
        for img in images:
            texts.extend(e_ocr.get_text(image))

        LPs = [p if re.match(plate, p) else None for p in texts]
        print(im, ':', end=' ')
        print(texts)
        for p in LPs:
            if p is not None:
                print(p, end=' ')
        print()
