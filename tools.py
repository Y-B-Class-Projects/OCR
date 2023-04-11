import datetime

import cv2
import imutils
import numpy as np
from PIL import Image
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
import cupy as cp

from easyocr_imp import EasyOCR


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class OCRInstances(metaclass=Singleton):
    def __init__(self):
        self.instances = EasyOCR(['en'], rec_network='best_accuracy')

    def get_instances(self):
        return self.instances


# get grayscale image
def get_grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


def remove_red(image):
    common_pixel = find_most_common_pixel(image)
    common_pixel.reverse()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    min_s = [50]
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
        image_temp[mask3 == 0] = common_pixel
        image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
        # _, image_temp = cv2.threshold(image_temp, 200, 255, cv2.THRESH_OTSU)
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
    ret = []
    if image_path.endswith('.tif') or image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith(
            '.jpeg'):
        image = cv2.imread(image_path)
        image = cut_image(image)
        height, width, channels = image.shape
        new_width = 516  # pixels
        new_height = int(height * (new_width / width))
        image = cv2.resize(image, (new_width, new_height))
        gray = get_grayscale(image)

        ret.append(image)
        ret.append(gray)
        # ret.append(thresholding(gray))
        ret += remove_red(image)

    return ret


def cut_image(image):
    e_ocr = OCRInstances().get_instances()
    height, width, _ = image.shape

    boxs = e_ocr.get_boxes(image)  # Convert cupy array to numpy array for easyocr
    all_points = np.array(boxs).reshape(-1, 8)
    # Get the minimum and maximum x and y coordinates
    min_x = int(max(np.min(all_points[:, [0, 2, 4, 6]]) - 2, 0))
    min_y = int(max(np.min(all_points[:, [1, 3, 5, 7]]) - 2, 0))
    max_x = int(min(np.max(all_points[:, [0, 2, 4, 6]]) + 2, width))
    max_y = int(min(np.max(all_points[:, [1, 3, 5, 7]]) + 2, height))

    image_new = image[min_y:max_y, min_x:max_x]

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


def show_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im_pil.show()


def find_most_common_pixel(image):
    histogram = {}  # Dictionary keeps count of different kinds of pixels in image

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    for x in range(image.width):
        for y in range(image.height):
            pixel_val = get_pixel_value(image.getpixel((x, y)))
            if pixel_val in histogram:
                histogram[pixel_val] += 1  # Increment count
            else:
                histogram[pixel_val] = 1  # pixel_val encountered for the first time

    mode_pixel_val = max(histogram, key=histogram.get)  # Find pixel_val whose count is maximum
    return get_rgb_values(mode_pixel_val)  # Returna a list containing RGB Value of the median pixel


def get_rgb_values(pixel_value):
    red = pixel_value % 256
    pixel_value //= 256
    green = pixel_value % 256
    pixel_value //= 256
    blue = pixel_value
    return [red, green, blue]


def get_pixel_value(pixel):
    return pixel[0] + 256 * pixel[1] + 256 * 256 * pixel[2]

