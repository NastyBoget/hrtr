import cv2
import numpy as np
from PIL import Image

from src.dataset.preprocessing.binarization import Binarizer

binarizer = Binarizer()


def preprocess(img: Image.Image) -> Image.Image:
    """
    :param img: pil image with white background
    :return: preprocessed image
    """
    img = np.array(img)
    if len(img.shape) < 3:
        rgb = False
        img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
    else:
        rgb = True
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img = binarizer.binarize(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if not rgb else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
