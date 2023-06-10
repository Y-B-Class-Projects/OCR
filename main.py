from tools import process_image

if __name__ == '__main__':

    from easyocr_imp import EasyOCR

    e_ocr = EasyOCR(['en'], rec_network='best_accuracy')

    im = "data/12F9-9883.jpg"
    image, gray, thresh = process_image(im)
    images = [image]
    # images.extend(remove_red(image))
    texts = []
    for img in images:
        texts.extend(e_ocr.get_text(image))

    for text in texts:
        print(text)

