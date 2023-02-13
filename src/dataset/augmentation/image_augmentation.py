import random

import cv2
import numpy as np
import skimage.exposure
from PIL import Image
from augmixations import HandWrittenBlot
from numpy.random import default_rng

from dataset.preprocessing.binarization import Binarizer


binarizer = Binarizer()


def augment(img: Image.Image, opt: any) -> Image.Image:
    """
    :param img: pil image with white background
    :return: augmented image
    """
    img = np.array(img)
    if len(img.shape) < 3:
        rgb = False
        img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    else:
        rgb = True
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if opt.preprocessing:
        img = binarizer.binarize(img)
    augment_probability = 0.3

    # Text augmentation
    img = change_width(img, proportion=random.uniform(0.5, 2.))
    img = blur(img, max_blur=random.randint(1, 7)) if random.random() < augment_probability else img
    img = resize_down(img) if random.random() < augment_probability else img
    img = resize_up(img) if random.random() < augment_probability else img
    img = erode(img) if random.random() < augment_probability else img
    img = dilate(img) if random.random() < augment_probability else img

    # Background augmentation
    img = add_blot(img, blots_num=random.randint(0, 3))
    img = add_blurred_stains(img, max_color=random.randint(100, 150), light=bool(random.randint(0, 1))) \
        if random.random() < augment_probability else img
    img = add_cut_characters(img, under=bool(random.randint(0, 1))) if random.random() < augment_probability else img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if not rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


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


def add_blot(img: np.ndarray, blots_num: int) -> np.ndarray:
    """
    The augmentation was taken from https://github.com/ai-forever/StackMix-OCR
    :param img: image to change
    :param blots_num: number of blots to draw on the picture
    :return: image with blots
    """
    b = HandWrittenBlot(
        {'x': (None, None), 'y': (None, None), 'h': (int(img.shape[0] * 0.1), int(img.shape[0] * 0.5)), 'w': (int(img.shape[1] * 0.1), int(img.shape[1] * 0.2))},  # noqa
        {'incline': (10, 50), 'intensivity': (0.75, 0.75), 'transparency': (0.05, 0.4), 'count': blots_num}
    )
    return b.apply(img)


def fill_gradient(img: np.ndarray, color: int, rotate: int = 0, light: bool = False) -> np.ndarray:
    """
    Fill the background with the gradient color
    :param img: image where background is 255 and text is black 0
    :param color: color of the lines to draw
    :param rotate: number of times to rotate the background
    :param light: make the picture lighter with the new background
    :return: image with gradient background
    """
    background_img = np.ones((img.shape[1], img.shape[1], 3), dtype=np.uint8) * 255
    gradient_mask = np.rot90(np.repeat(np.tile(np.linspace(1, 0, background_img.shape[0]),
                                               (background_img.shape[0], 1))[:, :, np.newaxis], 3, axis=2), rotate)
    background_img[:, :, :] = gradient_mask * background_img + (1 - gradient_mask) * color
    background_img = 255 - cv2.resize(background_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

    if light:
        return cv2.add(img, background_img)
    else:
        img = cv2.add(255 - img, background_img)
        return 255 - img


def add_stains(img: np.ndarray, color: int, light: bool = False) -> np.ndarray:
    """
    Adds stains with sharp borders to the picture
    :param img: image to change
    :param color: color of the stains
    :param light: make the picture lighter with the new background
    :return: image with stains
    """
    rng = default_rng(seed=random.randint(0, 1000))
    noise = rng.integers(0, 255, (img.shape[0], img.shape[1]), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255))
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask, mask, mask])
    result = np.where(mask > 0, 255 - color, 0).astype(np.uint8)

    if light:
        return cv2.add(img, result)
    else:
        img = cv2.add(255 - img, result)
        return 255 - img


def add_blurred_stains(img: np.ndarray, max_color: int, light: bool = False) -> np.ndarray:
    """
    Adds stains with sharp borders to the picture
    :param img: image to change
    :param max_color: maximum value of the stains color
    :param light: make the picture lighter with the new background
    :return: image with stains
    """
    rng = default_rng(seed=random.randint(0, 1000))
    noise = rng.integers(0, 255, (img.shape[0], img.shape[1]), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)

    background_img = cv2.merge([blur, blur, blur]) / 255.
    background_img = ((background_img - background_img.min()) * max_color / (background_img.max() - background_img.min())).astype(np.uint8)

    if light:
        return cv2.add(img, background_img)
    else:
        img = cv2.add(255 - img, background_img)
        return 255 - img


def add_cut_characters(img: np.ndarray, under: bool = True) -> np.ndarray:
    """
    Add shifted higher or lower parts of the given word to up or down of the word
    :param img: image to change
    :param under: add cut characters under the word if True, else above
    :return: image with cut characters
    """
    img = 255 - img
    y = int(0.3 * img.shape[0])
    cut_img = img[:y, :] if under else img[-y:, :]
    cut_img = np.roll(cut_img, random.randint(0, cut_img.shape[1]), axis=1)

    if under:
        img[-y:, :] = cv2.add(img[-y:, :], cut_img)
    else:
        img[:y, :] = cv2.add(img[:y, :], cut_img)
    return 255 - img
