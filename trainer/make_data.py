import os
import random

from imgaug import augmenters as iaa
import cv2
import numpy as np
import rstr
import pandas as pd
import secrets

from tqdm import tqdm
from trdg.generators import (
    GeneratorFromStrings,
)

IMAGE_COUNT = 150000
TEST_PERCENT = 0.1
ALL_CHARS = []

with open('vi_char.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    chars = []
    for line in lines:
        chars.append(line.strip())

# ALL_CHARS = chars
ALL_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:()'
VI_WORDS = ['Tai', 'trong', 'So', 'cho', 'ngoi', 'Dung', 'tich', 'nguoi', 'duoc', 'phep', 'cho']
VI_WORDS_ORIGINAL = ['Tải', 'trọng', 'Số', 'chỗ', 'ngồi', 'Dung', 'tích', 'người', 'được', 'phép', 'chở']
EN_WORDS = ['Seat capacity', 'Capacity', 'Sit']


def create_string_test():
    return ''.join(secrets.choice(ALL_CHARS) for _ in range(random.randint(4, 10)))


def create_strings(func):
    return [func() for _ in range(IMAGE_COUNT)]


def random_lpr_string():
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


def random_vi_string():
    st = secrets.choice(VI_WORDS)
    if secrets.randbelow(2):
        st = '(' + st
    if secrets.randbelow(2):
        st += ')'
    if secrets.randbelow(2):
        st += ':'

    return st


def random_en_string():
    st = secrets.choice(EN_WORDS)
    if secrets.randbelow(2):
        st = '(' + st
    if secrets.randbelow(2):
        st += ')'
    if secrets.randbelow(2):
        st += ':'

    return st


def random_vi_random_string():
    st = rstr.xeger(r'[A-Z]{2,3} \d{2,3} [A-Z]{2,3}')
    if secrets.randbelow(2):
        st = '(' + st
    if secrets.randbelow(2):
        st += ')'
    if secrets.randbelow(2):
        st += ':'

    return st


def get_generator(def_random_string, fonts, images_count):
    return GeneratorFromStrings(strings=create_strings(def_random_string),
                                fonts=fonts,
                                image_dir='background',
                                skewing_angle=5,
                                random_blur=False,
                                random_skew=True,
                                distorsion_type=0,
                                background_type=3,
                                count=images_count,
                                size=100,
                                character_spacing=0,
                                margins=(10, 10, 10, 10),
                                text_color="#2d3238")


def create_data(count, dir_path, generators):
    df = pd.DataFrame(columns=['filename', 'words'])
    for i in tqdm(range(count)):
        generator = secrets.choice(generators)
        image, words = generator.next()
        if words in VI_WORDS_ORIGINAL:
            words = VI_WORDS[VI_WORDS_ORIGINAL.index(words)]
        file_name = str(i).zfill(5) + '.png'
        file_path = os.path.join(dir_path, file_name)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
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

    df.to_csv(dir_path + '/labels.csv', encoding='utf-8-sig', header=['filename', 'words'], index=False)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(ROOT_DIR, 'all_data', 'test')
data_dir = os.path.join(ROOT_DIR, 'all_data', 'train')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

lpr_fonts_dir = ['fonts/FreeSerif.ttf', 'fonts/FreeSerifBold.ttf']
vi_fonts_dir = ['fonts/FreeSerif.ttf', 'fonts/FreeSerifBold.ttf']
en_fonts_dir = ['fonts/FreeSerifItalic.ttf', 'fonts/FreeSerifBoldItalic.ttf']

lpr_data_generator = get_generator(random_lpr_string, lpr_fonts_dir, IMAGE_COUNT)
# vi_data_generator = get_generator(random_vi_string, vi_fonts_dir, IMAGE_COUNT)
en_data_generator = get_generator(create_string_test, en_fonts_dir, IMAGE_COUNT)

create_data(int(IMAGE_COUNT * (1 - TEST_PERCENT)), data_dir, [lpr_data_generator])
create_data(int(IMAGE_COUNT * TEST_PERCENT), val_dir, [lpr_data_generator])
