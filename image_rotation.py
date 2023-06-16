import numpy as np
from cv2 import cv2
from scipy.spatial import ConvexHull
import OcrEngines


def minimum_bounding_rectangle(points):
    pi2 = np.pi / 2.
    hull_points = points[ConvexHull(points).vertices]
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    rot_points = np.dot(rotations, hull_points.T)
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def smart_deskew(image):
    old_image_height, old_image_width, channels = image.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = old_image_width + 500
    new_image_height = old_image_height + 500
    color = (200, 200, 200)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = image
    image = result

    width, height, _ = image.shape

    e_ocr = OcrEngines.EasyOcrImp.EasyOCR(['en'])
    boxs = e_ocr.get_boxes(image)  # Convert cupy array to numpy array for easyocr
    all_points = np.array(boxs).reshape(-1, 2)

    bounding_points = minimum_bounding_rectangle(all_points)

    bounding_points = sorted(bounding_points, key=lambda t: t[1])
    bounding_points = sorted(bounding_points, key=lambda t: t[0])
    temp = bounding_points[2]
    bounding_points[2] = bounding_points[3]
    bounding_points[3] = temp

    pt_A, pt_B, pt_C, pt_D = bounding_points

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32(bounding_points)
    output_pts = np.float32([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return out
