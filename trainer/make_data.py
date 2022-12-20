import os

import cv2
import numpy as np
import rstr
import pandas as pd

from trdg.generators import (
    GeneratorFromRandom,
    GeneratorFromStrings,
)

from tools import thresholding

image_count = 10
val_percent = 0.2


def create_strings():
    return [random_string() for _ in range(image_count)]


def random_string():
    reg_s = r"([0-9][0-9])(([A-Z][A-Z]?-)|(-[A-Z]([0-9]|[A-Z])))(([0-9][0-9][0-9][0-9])|([0-9][0-9][0-9]\.[0-9][0-9]))"
    return rstr.xeger(reg_s)


def create_data(count, dir_path, generator):
    df = pd.DataFrame(columns=['filename', 'words'])
    for i in range(count):
        image, words = generator.next()
        file_name = str(i).zfill(5) + '.png'
        file_path = os.path.join(dir_path, file_name)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=0, sigmaY=0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
        image = clahe.apply(image)
        _, image = cv2.threshold(image, thresh=165, maxval=255, type=cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        cv2.imwrite(file_path, image)
        df = df.append({'filename': file_name, 'words': words}, ignore_index=True)

    df.to_csv(dir_path + '/labels.csv', index=False)


fonts_dir = ['fonts/Scheherazade-Bold.ttf', 'fonts/Scheherazade-Regular.ttf']
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(ROOT_DIR, 'all_data', 'test')
data_dir = os.path.join(ROOT_DIR, 'all_data', 'train')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Generate 1000 synthetic images of text in the OCR-A font
data_generator = GeneratorFromStrings(strings=create_strings(),
                                      fonts=fonts_dir,
                                      image_dir='background',
                                      background_type=4,  # 4 for ages
                                      skewing_angle=5,
                                      random_blur=False,
                                      random_skew=True,
                                      distorsion_type=0,  # Random
                                      count=image_count,
                                      size=64,
                                      character_spacing=0,
                                      fit=True)

create_data(int(image_count * (1 - val_percent)), data_dir, data_generator)
create_data(int(image_count * val_percent), val_dir, data_generator)
