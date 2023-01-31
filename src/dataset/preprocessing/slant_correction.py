"""Refactored copy of https://github.com/sueiras/handwriting_recognition_thesis """
import numpy as np
import cv2


def slant_angle(img: np.ndarray, threshold_up: int = 100, threshold_down: int = 100) -> float:
    """
    Calculate slant angle for cursive text
        - Check the upper neighbors of pixels with left blank
        - Use after doing a contrast enhancement
        - Use after having corrected the slope of the line
    :param img: Grayscale image, background value 0 and black value 255
    :param threshold_up: threshold of gray to decide that something is black
    :param threshold_down: gray threshold to decide that something is white
    :return: slant angle in radians
    """
    angle = []
    C = 0
    L = 0
    R = 0
    for w in range(1, img.shape[1] - 1):
        for h in range(2, img.shape[0] - 1):
            if np.mean(img[h, w]) > threshold_up and np.mean(img[h, w - 1]) < threshold_down:
                if np.mean(img[h - 1, w - 1]) > threshold_up:  # if top left is black
                    L += 1
                    angle += [-45 * 1.25]
                elif np.mean(img[h - 1, w]) > threshold_up:  # if top center is black
                    C += 1
                    angle += [0]
                elif np.mean(img[h - 1, w + 1]) > threshold_up:  # if top right is black
                    R += 1
                    angle += [45 * 1.25]
    return np.arctan2((R - L), (L + C + R))


def correct_slant(img: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Fix slant angle for cursive text.
    :param img: grayscale image, background value 255 and black value 0
    :param threshold: gray threshold to decide that something is white or black
    :return: fixed image
    """
    angle = slant_angle(img, threshold_up=threshold, threshold_down=threshold)
    img = 255 - img

    # Add blanks in laterals to compensate the shear transformation cut
    if angle > 0:
        img = np.concatenate([img, np.zeros([img.shape[0], int(img.shape[0] * angle), 3])], axis=1)
    else:
        img = np.concatenate([np.zeros([img.shape[0], int(img.shape[0] * (-angle)), 3]), img], axis=1)

    M = np.float32([[1, -angle, 0], [0, 1, 0]])
    img2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return 255 - img2
