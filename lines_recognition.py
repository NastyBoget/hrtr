from typing import List

import cv2
import numpy as np


def recognize_lines(img: np.ndarray, t: int = 0.3) -> List[int]:
    """
    n - height of the image in pixels
    h - projection profile (length = n)
    x - [1..n] - domain of h
    a - set of all checked points
    b - set of intervals corresponding to peaks of the projection profile - set of "lines"
    b_c - complement of b
    r - range of width of the peak in a given iteration
    s - set of separators between lines
    alpha = 0.1
    t - threshold (we found it manually)
    t_a - absolute value of the threshold in a given iteration
    :param img: input text image
    :return: list of separators between lines
    """
    alpha = 0.1
    t = t
    h = img.sum(axis=1).sum(axis=1)
    max_h = np.max(h)
    h = max_h - h  # we get inverse values of projections
    max_h = np.max(h)
    n = len(h)
    x = np.argsort(h)[::-1]
    a = set()
    b = []
    for x_i in x:
        if h[x_i] <= alpha * max_h:
            break
        t_a = t * h[x_i]
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

        r = (x1, x2)
        if r not in a:
            b.append(r)
            a.add(r)

    b = sorted(b)
    b_c = []
    s = []
    for i in range(len(b) - 1):
        x1, x2 = b[i]
        x1_next, x2_next = b[i + 1]
        if x2 < x1_next:
            b_c.append((x2, x1_next))
            s.append(x2 + np.argmin(h[x2:x1_next]))

    return s


if __name__ == "__main__":
    img = cv2.imread("data/lines_detection/3.png")

    print(recognize_lines(img))
