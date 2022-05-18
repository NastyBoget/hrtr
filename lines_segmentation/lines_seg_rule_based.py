from typing import List


def sort_bboxes(bboxes: List[tuple]) -> List[List[tuple]]:
    lines_list = []
    bboxes = sorted(bboxes, key=lambda x: x[1])
    for bbox in bboxes:
        if len(lines_list) == 0:
            lines_list.append([bbox])
            continue
        iou_threshold = 0.4
        prev_bbox = lines_list[-1][-1]
        min_y1, max_y1 = min(bbox[1], prev_bbox[1]), max(bbox[1], prev_bbox[1])
        min_y2, max_y2 = min(bbox[3], prev_bbox[3]), max(bbox[3], prev_bbox[3])
        threshold = (min_y2 - max_y1) / (max_y2 - min_y1)
        if threshold >= iou_threshold:
            lines_list[-1].append(bbox)
        else:
            lines_list.append([bbox])
    lines_list = [sorted(line) for line in lines_list]
    return lines_list
