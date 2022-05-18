from typing import List, Tuple

import cv2
import numpy as np


def find_lines_separators(b: List[Tuple[int, int]], h: np.ndarray) -> List[int]:
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


def sort_bboxes(img: np.ndarray, bboxes: List[tuple]) -> List[List[tuple]]:
    x_min, x_max = min([bbox[0] for bbox in bboxes]), max([bbox[2] for bbox in bboxes])
    y_min, y_max = min([bbox[1] for bbox in bboxes]), max([bbox[3] for bbox in bboxes])
    line_separators = recognize_lines(img[y_min:y_max, x_min:x_max])
    if len(line_separators) == 0:
        return [bboxes]
    lines_dict = {}
    if line_separators[0] > 0:
        lines_dict[(y_min, y_min + line_separators[0])] = []
    for ind in range(len(line_separators) - 1):
        lines_dict[(y_min + line_separators[ind], y_min + line_separators[ind + 1])] = []
    if y_min + line_separators[-1] < y_max:
        lines_dict[(y_min + line_separators[-1], y_max)] = []

    for bbox in bboxes:
        max_iou, line_key = 0, None
        for y1, y2 in lines_dict.keys():
            min_y1, max_y1 = min(bbox[1], y1), max(bbox[1], y1)
            min_y2, max_y2 = min(bbox[3], y2), max(bbox[3], y2)
            iou = (min_y2 - max_y1) / (max_y2 - min_y1)
            if iou > max_iou:
                max_iou = iou
                line_key = (y1, y2)
        assert (line_key is not None)
        lines_dict[line_key].append(bbox)

    lines_list = []
    for key in sorted(lines_dict.keys()):
        value = lines_dict[key]
        if len(value) > 0:
            lines_list.append(sorted(value))
    return lines_list


if __name__ == "__main__":
    img = cv2.imread("../data/lines_detection/3.png")

    print(recognize_lines(img))
