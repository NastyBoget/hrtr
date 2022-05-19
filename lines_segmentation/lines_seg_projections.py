from typing import List

import numpy as np
from scipy.signal import savgol_filter, find_peaks

from lines_segmentation.binarization import binarize


def find_lines_separators(img: np.ndarray) -> np.ndarray:
    binarized_img = binarize(img)
    projection = binarized_img.sum(axis=1).sum(axis=1)
    projection_smoothed = savgol_filter(projection, 70, 3)
    projection_smoothed = savgol_filter(projection_smoothed, 140, 3)
    peaks, _ = find_peaks(projection_smoothed, height=0)
    return peaks


def sort_bboxes(img: np.ndarray, bboxes: List[tuple]) -> List[List[tuple]]:
    if len(bboxes) == 0:
        return [bboxes]
    x_min, x_max = min([bbox[0] for bbox in bboxes]), max([bbox[2] for bbox in bboxes])
    y_min, y_max = min([bbox[1] for bbox in bboxes]), max([bbox[3] for bbox in bboxes])
    line_separators = find_lines_separators(img[y_min:y_max, x_min:x_max])
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
