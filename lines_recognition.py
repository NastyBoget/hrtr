from typing import List, Tuple

import cv2
import numpy as np


def recognize_lines(img: np.ndarray, t: float = 0.3) -> List[int]:
    """
    n - height of the image in pixels
    h - projection profile (length = n)
    x - [1..n] - domain of h
    a - set of all checked points
    b - set of intervals corresponding to peaks of the projection profile - set of "lines"
    r - range of width of the peak in a given iteration
    s - set of separators between lines
    alpha = 0.1
    t_a - absolute value of the threshold in a given iteration
    :param t: threshold (we found it manually)
    :param img: input text image
    :return: list of separators between lines
    """
    alpha = 0.1
    h = img.sum(axis=1).sum(axis=1)
    max_h = np.max(h)
    h = max_h - h  # we get inverse values of projections
    max_h = np.max(h)
    n, x, a, b = len(h), np.argsort(h)[::-1], set(), []
    for x_i in x:
        if h[x_i] <= alpha * max_h:
            break
        t_a = t * h[x_i]
        r = find_interval(h, n, t_a, x_i)
        if r not in a:
            b.append(r)
            a.add(r)

    s = find_lines_separators(b, h)
    return s


def find_lines_separators(b: List[Tuple[int, int]], h: np.ndarray):
    b = sorted(b)
    s = []
    for i in range(len(b) - 1):
        x1, x2 = b[i]
        x1_next, x2_next = b[i + 1]
        if x2 < x1_next:
            s.append(x2 + np.argmin(h[x2:x1_next]))
    return s


def find_interval(h: np.ndarray, n: int, t_a: float, x_i: int) -> Tuple[int, int]:
    x1 = x_i
    while x1 > 0:
        if h[x1] < t_a:
            break
        x1 -= 1
    x2 = x_i
    while x2 < n - 1:
        if h[x2] < t_a:
            break
        x2 += 1
    return x1, x2


if __name__ == "__main__":
    img = cv2.imread("data/lines_detection/3.png")

    print(recognize_lines(img))
