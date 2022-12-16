import math
from typing import Tuple

import numpy as np
from scipy import ndimage
from sklearn import linear_model


def detect_baseline(img: np.ndarray, threshold: int = 20) -> Tuple[int, int, int, float]:
    """
    detect baseline
    :param img: img to find a baseline
    :param threshold: TODO
    :returns: TODO
    """

    low = []
    for w in range(1, img.shape[1] - 1):
        if np.max(img[:, w]) > threshold:
            for h in range(img.shape[0] - 5, 0, -1):
                if img[h, w] > threshold:
                    low += [[h, w]]
                    break
    points_lower = np.array(low)

    # Robust outliers regression
    x = points_lower[:, 1].reshape(points_lower.shape[0], 1)
    y = points_lower[:, 0].reshape(points_lower.shape[0], 1)
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(x, y)
    y0 = model_ransac.predict(x[0].reshape(1, -1))[0][0]
    y1 = model_ransac.predict(x[-1].reshape(1, -1))[0][0]
    y_mean = model_ransac.predict(np.array([img.shape[1] / 2]).reshape(1, -1))
    angle = np.arctan((y1 - y0) / (x[-1] - x[0])) * (180 / math.pi)

    return y0, y1, int(y_mean), angle[0]


def crop_borders(img: np.ndarray) -> np.ndarray:
    """
    Crop borders
    """
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(img)
    if len(true_points) > 0:
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        img = img[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
                  top_left[1]:bottom_right[1] + 1]  # inclusive
    return img


def rescale(img: np.ndarray, threshold: int = 20) -> np.ndarray:
    img[img < 0] = 0
    img2 = np.array((img - np.min(img)) * (255 / (np.max(img)-np.min(img))))
    img2[img2 < threshold] = 0
    return img2


def correct_line_inclination(img: np.ndarray) -> np.ndarray:
    # Detect baseline to correct inclination
    y0, y1, y_mean, angle = detect_baseline(img)

    # Correct inclination with lower angle
    img_out = ndimage.rotate(img, angle)
    img_out = rescale(img_out)

    img_out = crop_borders(img_out)
    return img_out
