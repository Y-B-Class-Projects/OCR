import os
import random

import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import rstr
import pandas as pd
from scipy.stats._multivariate import random_correlation_gen

from trdg.generators import (
    GeneratorFromRandom,
    GeneratorFromStrings,
)

import tools
from tools import thresholding

image_count = 30000
val_percent = 0.2


def create_strings():
    return [random_string() for _ in range(image_count)]


def random_string():
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

    return rstr.xeger(plate)


def create_data(count, dir_path, generator):
    df = pd.DataFrame(columns=['filename', 'words'])
    for i in range(count):
        image, words = generator.next()
        file_name = str(i).zfill(5) + '.png'
        file_path = os.path.join(dir_path, file_name)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image_np = np.asarray(image)
        # Define the list of augmentations to apply
        augmentations = [
            iaa.Resize({"height": (0.8, 1.0), "width": (0.5, 1.0)}),
            iaa.Cutout(nb_iterations=1, fill_mode="constant", cval=255, size=0.05),
            iaa.Dropout(p=(0, 0.01)),  # randomly set pixels to zero
            iaa.GaussianBlur(sigma=(0.0, 1.5)),  # blur images
            iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # add Gaussian noise with a scale of 0 to 0.5*255
            iaa.PerspectiveTransform(scale=(0.01, 0.03)),
            iaa.Multiply(mul=(0.5, 1.5))  # multiply the image by a value ranging from 0.5 to 1.5
        ]

        # Define an imgaug augmentation pipeline using the list of augmentations
        seq = iaa.Sequential(augmentations)
        augmented_image = seq.augment_image(image_np)
        if random.random() <= 0.5:
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(file_path, augmented_image)
        df.loc[len(df.index)] = [file_name, words]

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
                                      skewing_angle=5,
                                      random_blur=False,
                                      random_skew=True,
                                      distorsion_type=0,
                                      background_type=3,
                                      count=image_count,
                                      size=100,
                                      character_spacing=0,
                                      margins=(10, 10, 10, 10),
                                      text_color="#2d3238")

create_data(int(image_count * (1 - val_percent)), data_dir, data_generator)
create_data(int(image_count * val_percent), val_dir, data_generator)
