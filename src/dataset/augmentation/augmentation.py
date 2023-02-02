import random

import numpy as np

from src.dataset.augmentation.background_augmentation import add_blot, fill_gradient, draw_lines, add_noise, add_stains, add_blurred_stains, \
    add_cut_characters
from src.dataset.augmentation.text_augmentation import change_width, blur, erode, dilate, resize_up, resize_down


def augment(img: np.ndarray) -> np.ndarray:
    """
    :param img: image with white background and 3 dimensions
    :return: list of augmented images
    """
    augment_probability = 0.2

    # Text augmentation
    img = change_width(img, proportion=random.uniform(0.5, 2.))
    img = blur(img, max_blur=random.randint(1, 7)) if random.random() < augment_probability else img
    img = resize_down(img) if random.random() < augment_probability else img
    img = resize_up(img) if random.random() < augment_probability else img
    img = erode(img) if random.random() < augment_probability else img
    img = dilate(img) if random.random() < augment_probability else img

    # Background augmentation
    img = draw_lines(img, color=random.randint(50, 200), thickness=random.randint(1, 3), lines_type=random.randint(0, 1),
                     line_width=random.randint(2, 3)) if random.random() < augment_probability else img
    img = add_blot(img, blots_num=random.randint(0, 3))
    img = fill_gradient(img, color=random.randint(100, 150), rotate=random.randint(0, 3), light=bool(random.randint(0, 1))) \
        if random.random() < augment_probability else img
    img = add_noise(img, max_dots=random.randint(0, 100), min_color=0, max_color=100)
    img = add_stains(img, color=random.randint(100, 150), light=bool(random.randint(0, 1))) if random.random() < augment_probability else img
    img = add_blurred_stains(img, max_color=random.randint(100, 150), light=bool(random.randint(0, 1))) \
        if random.random() < augment_probability else img
    img = add_cut_characters(img, under=bool(random.randint(0, 1))) if random.random() < augment_probability else img

    return img
