import cv2
import easyocr

from tools import get_grayscale, thresholding

reader = easyocr.Reader(['en'], recog_network='best_accuracy')  # this needs to run only once to load the model into memory


image = cv2.imread('data/LD_05.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

result = reader.readtext(image)

for res in result:
    print(res[1])

