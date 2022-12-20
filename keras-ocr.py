import keras_ocr
import matplotlib.pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

images = [
    keras_ocr.tools.read(img) for img in ['data/LD_01.jpg', 'data/LD_02.jpg']
]

prediction_groups = pipeline.recognize(images)

predicted_image = prediction_groups[0]
for text, box in predicted_image:
    print(text)
