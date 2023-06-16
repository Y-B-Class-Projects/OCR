import numpy as np
import cv2

image = cv2.imread('data/29L5-120.55.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# lower red
lower_red = np.array([0, 25, 50])
upper_red = np.array([10, 255, 255])

# upper red
lower_red2 = np.array([150, 25, 50])
upper_red2 = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.bitwise_not(mask)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask2 = cv2.bitwise_not(mask2)
mask3 = cv2.bitwise_and(mask, mask2)

image[mask3 == 0] = (218, 211, 194)

image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image1 = cv2.threshold(image1, 200, 255, cv2.THRESH_OTSU)

cv2.imshow('image', image)
cv2.imshow('image1', image1)

cv2.waitKey(0)
cv2.destroyAllWindows()
