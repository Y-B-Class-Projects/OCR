import datetime
import math
from math import ceil
from pathlib import Path
from prettytable import PrettyTable
import os
from tqdm import tqdm
import tools
from easyocr_imp import EasyOCR
from my_dictionary import MyDictionary
from our_ocr_engine import OurOcrEngine
from tools import bcolors, is_license_plate
import OCR_engine


class OcrEngineBenchmark:
    def __init__(self, ocr_engine):  # type: (OCR_engine) -> None
        self.instance = ocr_engine
        self.results = MyDictionary()
        self.true_count = 0

    def get_ocr_results(self, image):
        return self.instance.get_text_with_prob(image)

    def add_result(self, text, confidence):
        self.results.add_item(text, confidence)

    def reset_results(self):
        self.results.reset()

    def run(self, processed_images, true_value):
        self.reset_results()
        for processed_image in processed_images:
            for result in self.get_ocr_results(processed_image):
                if result is not None:
                    text, confidence = result
                    if is_license_plate(text):
                        self.add_result(text, confidence)

        result = self.results.get_max_item()
        if result == true_value:
            self.true_count += 1
            return bcolors.OKGREEN + "`" + result + "`" + bcolors.ENDC
        else:
            return bcolors.FAIL + "`" + result + "`" + bcolors.ENDC


class Benchmark:
    def __init__(self):
        self.images_path = None
        self.ocr_engines = []  # type: [OcrEngineBenchmark]
        self.table = PrettyTable(['our_result', 'easyocr_result', 'True Value'])

    def add_engine(self, ocr_engine):
        self.ocr_engines.append(OcrEngineBenchmark(ocr_engine))

    def run(self, path):
        self.images_path = path
        for image in tqdm(os.listdir(self.images_path)):
            engines_results = []
            true_value = Path(image).stem  # get filename without extension
            processed_images = tools.process_image(self.images_path + image)
            for ocr_engine in self.ocr_engines:  # type: OcrEngineBenchmark
                engines_results.append(ocr_engine.run(processed_images, true_value))

            # add results to table
            engines_results.append(true_value)
            self.table.add_row(engines_results)

        rates = []
        for ocr_engine in self.ocr_engines:
            rates.append(math.ceil(ocr_engine.true_count / len(os.listdir(self.images_path))*100))
        rates.append("")
        self.table.add_row(rates)

    def print_table(self):
        print(self.table)


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # create benchmark
    benchmark = Benchmark()

    # create engines
    our_engine = OurOcrEngine(['en'])
    easyocr_engine = EasyOCR(['en'])

    # add the engines
    benchmark.add_engine(our_engine)
    benchmark.add_engine(easyocr_engine)

    benchmark.run('data/')

    benchmark.print_table()

    print("Total time:", datetime.datetime.now() - start_time)
