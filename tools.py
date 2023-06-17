import re
import OcrEngines
import cv2
import imutils
import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import OcrEngines.EasyOcrImp


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]




# get grayscale image
def get_grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


def remove_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    min_s = [0, 25, 50]
    rets = []
    for s in min_s:
        lower_red1 = np.array([0, s, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([150, s, 50])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1)
        mask = cv2.bitwise_not(mask)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask2 = cv2.bitwise_not(mask2)
        mask3 = cv2.bitwise_and(mask, mask2)
        image_temp = image.copy()
        image_temp[mask3 == 0] = (218, 211, 194)
        image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
        _, image_temp = cv2.threshold(image_temp, 200, 255, cv2.THRESH_OTSU)
        rets.append(image_temp)

    return rets


# thresholding
def thresholding(image):
    return cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# skew correction
def deskew(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold to get rid of extraneous noise
    thresh = threshold_otsu(image)
    normalize = image > thresh

    # gaussian blur
    blur = gaussian(normalize, 3)

    # canny edges in scikit-image
    edges = canny(blur)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)

    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90 - rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def process_image(image_path):
    image_path = image_path.lower()
    if image_path.endswith(('.tif', '.png', '.jpg', '.jpeg')):
        image = cut_image(image_path)
        height, width, channels = image.shape
        new_width = 500  # pixels
        new_height = int(height * (new_width / width))
        image = cv2.resize(image, (new_width, new_height))
        rotation_angle = deskew(image)
        image = imutils.rotate(image, angle=rotation_angle)
        gray = get_grayscale(image)
        without_red_01, without_red_02, without_red_03 = remove_red(image)

        return image, gray, without_red_02
    else:
        print('Invalid image format')
        return None, None, None


def cut_image(image_path):
    e_ocr = OcrEngines.EasyOcrImp.EasyOCR(['en'], rec_network='best_accuracy')
    image = cv2.imread(image_path)
    min_x, min_y, max_x, max_y = image.shape[0], image.shape[1], 0, 0
    boxs = e_ocr.get_boxes(image_path)
    for box in boxs:
        all_x = [point[0] for point in box]
        all_y = [point[1] for point in box]
        min_x = min(min_x, min(all_x))
        min_y = min(min_y, min(all_y))
        max_x = max(max_x, max(all_x))
        max_y = max(max_y, max(all_y))

    min_x = int(min_x) - 2
    min_y = int(min_y) - 2
    max_x = int(max_x) + 2
    max_y = int(max_y) + 2

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, image.shape[1])
    max_y = min(max_y, image.shape[0])

    image_with = 900
    image_height = int(image_with * (max_y - min_y) / (max_x - min_x))

    pts1 = np.float32([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    pts2 = np.float32([[0, 0], [0, image_height], [image_with, 0], [image_with, image_height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_new = cv2.warpPerspective(image, matrix, (image_with, image_height))

    return image_new


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# def remove_red(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # lower red
#     lower_red = np.array([0, 25, 50])
#     upper_red = np.array([10, 255, 255])
#     # upper red
#     lower_red2 = np.array([150, 25, 50])
#     upper_red2 = np.array([180, 255, 255])
#
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     mask = cv2.bitwise_not(mask)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     mask2 = cv2.bitwise_not(mask2)
#     mask3 = cv2.bitwise_and(mask, mask2)
#
#     image[mask3 == 0] = (218, 211, 194)
#
#     image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, image1 = cv2.threshold(image1, 200, 255, cv2.THRESH_OTSU)
#
#     return image1


def is_license_plate(txt):
    # regex for license plate
    plate = r"(\d{2}[A-Z]-\d{3}\.\d{2})|" \
            r"(\d{2}[A-Z]{2}-\d{3}\.\d{2})|" \
            r"(\d{2}-[A-Z]\d{4}\.\d{2})|" \
            r"(\d{2}-[A-Z]{2}\d{3}\.\d{2})|" \
            r"(\d{2}[A-Z]-\d{3})|" \
            r"(\d{2}[A-Z]{2}-\d{3})|" \
            r"(\d{2}-[A-Z]\d{5})|" \
            r"(\d{2}-[A-Z]{2}\d{4})|" \
            r"(\d{2}[A-Z]\d-\d{4})|" \
            r"(\d{2}[A-Z]\d-\d{3}\.\d{2})|"\
            r"(11111)|" \
            r"(\d{2}[A-Z]\d{3}\d{2})|" \
            r"(\d{2}[A-Z]{2}\d{3}\d{2})|" \
            r"(\d{2}[A-Z]\d{4}\d{2})|" \
            r"(\d{2}[A-Z]{2}\d{3}\d{2})|" \
            r"(\d{2}[A-Z]\d{3})|" \
            r"(\d{2}[A-Z]{2}\d{3})|" \
            r"(\d{2}[A-Z]\d{5})|" \
            r"(\d{2}[A-Z]{2}\d{4})|" \
            r"(\d{2}[A-Z]\d\d{4})|" \
            r"(\d{2}[A-Z]\d\d{3}\d{2})"
    return re.match(plate, txt)


if __name__ == '__main__':
    image_path = "C:\\Users\\IZuser\\PycharmProjects\\OCR\\data\\29T2-5553.jpg"
    images = process_image(image_path)
    original = cv2.imread(image_path)
    cv2.imshow("Original", original)
    cv2.waitKey(0)
    for i, image in enumerate(images):
        cv2.imwrite(str(i)+".jpg", image)
