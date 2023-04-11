import os
import random
from imgaug import augmenters as iaa
import cv2
import numpy as np
import rstr
import pandas as pd
from string import ascii_lowercase, ascii_uppercase

from tqdm import tqdm
from trdg.generators import (GeneratorFromStrings)
from unidecode import unidecode

IMAGE_COUNT = 150000
VALIDATION_SPLIT = 0.1

VI_UPPERCASE = ascii_uppercase + "ĐĂÂÊÔƠƯÀẰẦÈỀÌÒỒỜÙỪỲẢẲẨẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸÁẮẤÉẾÍÓỐỚÚỨÝẠẶẬẸỆỊỌỘỢỤỰỴ():"
VI_LOWERCASE = ascii_lowercase + "đăâêôơưàằầèềìòồờùừỳảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹáắấéếíóốớúứýạặậẹệịọộợụựỵ():"

EN_SAVE_WORDS = ['Seat capacity', 'capacity', 'Sit']
VI_SAVE_WORDS = ['Số người được phép chở',  # Number of people allowed to carry
                 'Dung tích',  # Capacity
                 'Số chỗ ngồi',  # Number of seats
                 ]
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_strings(create_strings_func, count=IMAGE_COUNT):
    return [create_strings_func() for _ in range(count)]


def vi_random_string():
    choice = random.randrange(0, 2)
    if choice == 0:
        n_words = random.randrange(1, 4)
        words = []
        for i in range(n_words):
            word = random.choice(VI_UPPERCASE) + ''.join(
                random.choice(VI_LOWERCASE) for _ in range(random.randrange(3, 10)))
            words.append(word)
        return ' '.join(words)
    elif choice == 1:
        return random.choice(VI_SAVE_WORDS)


def en_random_string():
    choice = random.randrange(0, 2)
    if choice == 0:
        chars = ascii_lowercase+":()"
        n_words = random.randrange(1, 4)
        words = []
        for i in range(n_words):
            word = ''.join(random.choice(chars) for _ in range(random.randrange(3, 10)))
            word = word.title()
            words.append(word)
        return ' '.join(words)
    elif choice == 1:
        word = random.choice(EN_SAVE_WORDS)
        if random.randrange(0, 2):
            word = '(' + word + '):'
        return word


def plate_random_string():
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


def augment_image(image):
    image_np = np.asarray(image)
    augmentations = [
        iaa.Resize({"height": (0.8, 1.0), "width": (0.5, 1.0)}),
        iaa.Cutout(nb_iterations=1, fill_mode="constant", cval=255, size=0.05),
        iaa.Dropout(p=(0, 0.01)),  # randomly set pixels to zero
        iaa.GaussianBlur(sigma=(0.0, 1.5)),  # blur images
        iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # add Gaussian noise with a scale of 0 to 0.5*255
        iaa.PerspectiveTransform(scale=(0.01, 0.03)),
        iaa.Multiply(mul=(0.5, 1.5))  # multiply the image by a value ranging from 0.5 to 1.5
    ]
    seq = iaa.Sequential(augmentations)
    augmented_image = seq.augment_image(image_np)
    if random.random() <= 0.5:
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)
    return augmented_image


def create_data(_generators, dir_path, count=IMAGE_COUNT):
    count_name = 0
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df = pd.DataFrame(columns=['filename', 'words'])
    for _ in tqdm(range(int(count/len(_generators)))):
        for generator in _generators:
            image, words = generator.next()
            file_name = str(count_name).zfill(5) + '.png'
            count_name += 1
            file_path = os.path.join(dir_path, file_name)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            augmented_image = augment_image(image)
            cv2.imwrite(file_path, augmented_image)
            words = unidecode(words)
            df.loc[len(df.index)] = [file_name, words]
    df.to_csv(dir_path + '/labels.csv', index=False)


def get_data_generator(random_string_func, fonts_dir, count=IMAGE_COUNT):
    return GeneratorFromStrings(strings=create_strings(random_string_func), fonts=fonts_dir, image_dir='background',
                                skewing_angle=5, random_blur=False, random_skew=True,
                                distorsion_type=0, background_type=3, count=count, size=100, character_spacing=0,
                                margins=(10, 10, 10, 10), text_color="#2d3238")


if __name__ == '__main__':

    en_fonts_dir = ['fonts/FreeSerifItalic.ttf', 'fonts/FreeSerifBoldItalic.ttf']
    vi_fonts_dir = ['fonts/FreeSerif.ttf', 'fonts/FreeSerifBold.ttf']
    plate_fonts_dir = ['fonts/FreeSerifBold.ttf']

    train_dir = os.path.join(ROOT_DIR, 'data', 'train')
    test_dir = os.path.join(ROOT_DIR, 'data', 'test')

    vi_generator = get_data_generator(vi_random_string, vi_fonts_dir)
    en_generator = get_data_generator(en_random_string, en_fonts_dir)
    plate_generator = get_data_generator(plate_random_string, plate_fonts_dir)

    generators = [vi_generator, en_generator, plate_generator]

    create_data(generators, train_dir, count=int(IMAGE_COUNT * (1 - VALIDATION_SPLIT)))
    create_data(generators, test_dir, count=int(IMAGE_COUNT * VALIDATION_SPLIT))
