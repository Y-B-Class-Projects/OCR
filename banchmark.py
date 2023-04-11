import datetime
import os
import re
from math import ceil

from PIL import Image
from cv2 import cv2
from pathlib import Path
from prettytable import PrettyTable
import os

import tools
from my_dictionary import MyDictionary
from tools import bcolors

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import easyocr

start_time = datetime.datetime.now()

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

our_results = MyDictionary()
easyocr_results = MyDictionary()

our_count = 0
easyocr_count = 0

our_reader = easyocr.Reader(['en'], recog_network='best_accuracy')
easyocr_reader = easyocr.Reader(['en'])

t = PrettyTable(['origin', 'gray', 'remove red_1', 'our_result', 'easyocr_result', 'True Value'])

for im in os.listdir('data'):
    true_val = Path(im).stem
    images = tools.process_image('data/' + im)

    table_line = []

    our_results.reset()
    easyocr_results.reset()

    for image in images:
        line = []
        results = our_reader.readtext(image)
        for res in results:
            if res is not None:
                _, txt, conf = res
                if re.match(plate, txt):
                    our_results.add_item(txt, conf)
                    line.append(txt)
        table_line.append(', '.join(line))

    our_result = our_results.get_max_item()
    if our_result == true_val:
        table_line.append(bcolors.OKGREEN + "`" + our_result + "`" + bcolors.ENDC)
        our_count += 1
    else:
        table_line.append(bcolors.FAIL + "`" + our_result + "`" + bcolors.ENDC)

    for image in images:
        results = easyocr_reader.readtext(image)
        for res in results:
            if res is not None:
                _, txt, conf = res
                if re.match(plate, txt):
                    easyocr_results.add_item(txt, conf)

    easyocr_result = easyocr_results.get_max_item()
    if easyocr_result == true_val:
        table_line.append(bcolors.OKGREEN + "`" + easyocr_result + "`" + bcolors.ENDC)
        easyocr_count += 1
    else:
        table_line.append(bcolors.FAIL + "`" + easyocr_result + "`" + bcolors.ENDC)

    table_line.append(true_val)
    t.add_row(table_line)

our_rate = str(ceil((our_count / len(os.listdir('data'))) * 100)) + "%"
easyocr_rate = str(ceil((easyocr_count / len(os.listdir('data'))) * 100)) + "%"
t.add_row(["", "", "", our_rate, easyocr_rate, ""])

print(t)

print("Total time:", datetime.datetime.now() - start_time)
