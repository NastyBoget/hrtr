import random
from typing import List

import cv2
import numpy as np


# THIS MODULE WORKS WITH INVERSE IMAGES


def move_img(img: np.ndarray) -> np.ndarray:
    pixels_move = 1 + int(random.random() * 10)
    img2 = np.zeros_like(img)
    img2[:, pixels_move:] = img[:, :-pixels_move]
    return img2


def resize_down(img: np.ndarray) -> np.ndarray:
    factor = 0.95 - random.random() / 4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = np.ones_like(img) * 0
    img2[(h_ini - h_fin) // 2:-(h_ini - h_fin) // 2, :w_fin] = img1
    return img2


def resize_up(img: np.ndarray) -> np.ndarray:
    factor = 1 + random.random() / 4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = img1[h_fin - h_ini:, :w_ini]
    return img2


def get_img_augmented(image_list: List[np.ndarray]) -> List[np.ndarray]:
    augmented_image_list = []
    for img in image_list:
        if len(img.shape) > 2:
            img = img[:, :, 0]

        # Move left
        img = move_img(img)

        # Skew
        if random.random() < 0.8:
            angle = (random.random() - 0.5) / 3.
            M = np.float32([[1, -angle, 0.5 * img.shape[0] * angle], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        #  Resize
        if random.random() < 0.4:
            img = resize_down(img)
        elif random.random() < 0.4:
            img = resize_up(img)

        #  Erode - dilate
        if random.random() < 0.3:
            img = cv2.erode(img, np.ones(2, np.uint8), iterations=1)
        elif random.random() < 0.3:
            img = cv2.dilate(img, np.ones(2, np.uint8), iterations=1)

        augmented_image_list.append(img)

    return augmented_image_list
