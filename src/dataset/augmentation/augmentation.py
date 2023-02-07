import random

import cv2
import numpy as np
from PIL import Image

from src.dataset.augmentation.image_augmentation import add_blurred_stains, add_cut_characters, add_blot, change_width, blur, erode, dilate, \
    resize_up, resize_down
from src.dataset.preprocessing.binarization import Binarizer

binarizer = Binarizer()


def augment(img: Image.Image) -> Image.Image:
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
