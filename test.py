from PIL import Image
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import tools
from easyocr_imp import EasyOCR

reader = EasyOCR(['en'], rec_network='best_accuracy')

image = tools.process_image("data/29B1-479.82.jpg")[0]
height, width, _ = image.shape

result = reader.get_data(image)
common_pixel = tools.find_most_common_pixel(image)
img_color = Image.new('RGB', (width, height), tuple(common_pixel))

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax2.imshow(img_color)

# Plot image with bounding boxes and recognized text
for bbox, _text, score in result:
    x1, y1 = bbox[0][0], bbox[0][1]
    x2, y2 = bbox[2][0], bbox[2][1]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    ax2.text(x1, y1 + (y2 - y1), _text, fontsize=12, color='black')
    rec = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=0.5)
    ax1.add_patch(rec)

plt.show()
