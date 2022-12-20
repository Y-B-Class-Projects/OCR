import math
import os

from PIL import Image
from matplotlib import pyplot as plt, patches


def main(x_scale, y_scale):
    for im in os.listdir('data'):
        if im.startswith('eng') and im.endswith('.tif'):
            image = Image.open(os.path.join('data', im))

            new_im = image.resize((int(image.size[0] * x_scale), (int(image.size[1] * y_scale))))
            new_im.save(os.path.join('data', f"resized_{im}"))

            box_file = open('data\\' + im.split('.')[0] + ".box", "r")
            new_box_file_name = 'data\\' + 'resized_' + im.split('.')[0] + ".box"

            print('from:', box_file.name, 'to:', new_box_file_name)

            with open(new_box_file_name, 'w+') as new_box_file:
                for line in box_file.readlines():
                    if len(line.split()) == 6:
                        c, x0, y0, x1, y1, d = line.split()

                        x0 = int(int(x0) * x_scale)
                        x1 = int(int(x1) * x_scale)
                        y0 = int(int(y0) * y_scale)
                        y1 = int(int(y1) * y_scale)

                        new_box_file.write(f"{c} {x0} {y0} {x1} {y1} {d}\n")


def show_box():
    for im in os.listdir('data'):
        if 'eng' in im and im.endswith('.tif'):
            image = Image.open(os.path.join('data', im))
            box_file = open('data\\' + im.split('.')[0] + ".box", "r")

            fig, ax = plt.subplots()
            ax.imshow(image)

            for line in box_file.readlines():
                if len(line.split()) == 6:
                    _, x0, y0, x1, y1, _ = line.split()

                    x0 = int(x0)
                    y0 = int(y0)
                    x1 = int(x1)
                    y1 = int(y1)

                    y0 = abs(y0 - image.size[1])
                    y1 = abs(y1 - image.size[1])

                    rect = patches.Rectangle((x0, y0), (x1 - x0), (y1 - y0), linewidth=1, edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
            # plt.show()


if __name__ == '__main__':
    main(0.6, 1.4)
    show_box()
