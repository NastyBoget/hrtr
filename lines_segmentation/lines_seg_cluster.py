from collections import defaultdict
from typing import List

from sklearn.cluster import DBSCAN


def find_lines_bboxes(bboxes: List[tuple]) -> List[tuple]:
    y_mean_list = []
    for bbox in bboxes:
        y1, y2 = bbox[1], bbox[3]
        y_mean_list.append(y1 + (y2 - y1) / 2)
    points = [[1, y] for y in y_mean_list]
    clusters = DBSCAN(min_samples=1, eps=30).fit_predict(points)

    lines_bboxes = []
    prev_cluster = None
    for bbox, line_idx in zip(bboxes, clusters):
        if line_idx != prev_cluster:
            prev_cluster = line_idx
            lines_bboxes.append([])
        lines_bboxes[-1].append(bbox)

    lines = []
    for line_bbox_list in lines_bboxes:
        max_x = max([bbox[2] for bbox in line_bbox_list])
        min_x = min([bbox[0] for bbox in line_bbox_list])
        max_y = max([bbox[3] for bbox in line_bbox_list])
        min_y = min([bbox[1] for bbox in line_bbox_list])
        lines.append((min_x, min_y, max_x, max_y))
    return lines


def is_in(bbox1: tuple, bbox2: tuple) -> bool:
    """
    check if bbox1 is inside bbox2
    bbox = (x1, y1, x2, y2)
    (x1, y1)
        ---------------
        |             |
        |    bbox     |
        |             |
        ---------------
                    (x2, y2)
    x1 < x2, y1 < y2
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    return x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2


def sort_bboxes(bboxes: List[tuple]) -> List[List[tuple]]:
    lines_bboxes = find_lines_bboxes(bboxes)
    lines_dict = defaultdict(list)
    for bbox in bboxes:
        for line_bbox in lines_bboxes:
            if is_in(bbox, line_bbox):
                lines_dict[line_bbox].append(bbox)

    lines_list = []
    for line_bbox in sorted(lines_dict.keys(), key=lambda x: x[1]):
        lines_list.append(sorted(lines_dict[line_bbox]))
    return lines_list
