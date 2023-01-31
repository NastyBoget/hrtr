import random

import cv2
import numpy as np


def change_width(img: np.ndarray, proportion: float) -> np.ndarray:
    """
    :param img: image to change
    :param proportion: parameter for multiplication of image's width
    :return: changed image
    """
    new_width = int(img.shape[1] * proportion)
    out_img = cv2.resize(img, (new_width, img.shape[0]), interpolation=cv2.INTER_AREA)
    return out_img


def blur(img: np.ndarray, max_blur: int) -> np.ndarray:
    return cv2.blur(img, (random.randint(1, max_blur), random.randint(1, max_blur)))


# The augmentations below were taken from https://github.com/sueiras/handwriting_recognition_thesis

def erode(img: np.ndarray) -> np.ndarray:
    img = cv2.erode(img, np.ones(2, np.uint8), iterations=1)
    return img


def dilate(img: np.ndarray) -> np.ndarray:
    img = cv2.dilate(img, np.ones(2, np.uint8), iterations=1)
    return img


def resize_down(img: np.ndarray) -> np.ndarray:
    factor = 0.95 - random.random() / 4.
    h_ini, w_ini = img.shape[0], img.shape[1]
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape[0], img1.shape[1]
    img2 = np.ones_like(img, dtype=np.uint8) * 255
    img2[(h_ini - h_fin) // 2:-(h_ini - h_fin) // 2, :w_fin] = img1
    return img2


def resize_up(img: np.ndarray) -> np.ndarray:
    factor = 1 + random.random() / 16.
    h_ini, w_ini = img.shape[0], img.shape[1]
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape[0], img1.shape[1]
    img2 = img1[h_fin - h_ini:, :w_ini]
    return img2
