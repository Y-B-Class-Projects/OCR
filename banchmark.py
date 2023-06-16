import datetime
from math import ceil
from pathlib import Path
from prettytable import PrettyTable
import os
from tqdm import tqdm
from OcrEngines.OurOCR import OurOcr
from tools import bcolors, is_license_plate, process_image
from OcrEngines.EasyOcrImp import EasyOCR
from OcrEngines.Pytesseract import Pytesseract
from OcrEngines import OCR_engine


class OcrResultDictionary:
    def __init__(self):
        self.dict = {}

    def add_item(self, key, val):
        if key in self.dict:
            prev_val = self.dict[key]
            val += prev_val
        self.dict[key] = val

    def get_max_item(self):
        if len(self.dict) > 0:
            return max(self.dict, key=self.dict.get)
        else:
            return "-------"

    def reset(self):
        self.dict = {}


class OcrEngineBenchmark:
    def __init__(self, ocr_engine):  # type: (OCR_engine) -> None
        self.instance = ocr_engine
        self.results = OcrResultDictionary()
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
        self.ocr_engines_names = []
        self.table = None

    def add_engine(self, ocr_engine, ocr_name):
        self.ocr_engines_names.append(ocr_name)
        self.ocr_engines.append(OcrEngineBenchmark(ocr_engine))

    def run(self, path):
        self.table = PrettyTable(self.ocr_engines_names + ["True Value"])
        self.images_path = path
        for image in tqdm(os.listdir(self.images_path)):
            engines_results = []
            true_value = Path(image).stem  # get filename without extension
            processed_images = process_image(self.images_path + image)
            for ocr_engine in self.ocr_engines:  # type: OcrEngineBenchmark
                engines_results.append(ocr_engine.run(processed_images, true_value))

            # add results to table
            engines_results.append(true_value)
            self.table.add_row(engines_results)

        rates = []
        for ocr_engine in self.ocr_engines:
            rates.append(ceil(ocr_engine.true_count / len(os.listdir(self.images_path)) * 100))
        rates.append("")
        self.table.add_row(rates)

    def print_table(self):
        print(self.table)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    # create benchmark
    benchmark = Benchmark()
    # create engines
    our_engine = OurOcr(['en'])
    easyocr_engine = EasyOCR(['en'])
    pytesseract_engine = Pytesseract()
    # add the engines
    benchmark.add_engine(our_engine, "Our Engine")
    benchmark.add_engine(easyocr_engine, "EasyOCR Engine")
    benchmark.add_engine(pytesseract_engine, "Pytesseract Engine")
    # run the benchmark
    benchmark.run('data/')
    benchmark.print_table()
    # print total time
    print("Total time:", datetime.datetime.now() - start_time)
