from typing import List

import numpy as np
from sklearn.cluster import DBSCAN


class LinesSegmenter:

    def sort_bboxes_by_clustering(self, bboxes: List[tuple]) -> List[List[tuple]]:
        """
        Sort list bboxes (x_top_left, y_top_left, x_bottom_right, y_bottom_right) into lines
        (x_top_left, y_top_left)
        |----------------------------------|
        |                                  |
        |----------------------------------|
                                    (x_bottom_right, y_bottom_right)

        Line1: bbox11, bbox12, ....
        Line2: bbox21, bbox22, ...
        .....
        """

        heights = [b[3] - b[1] for b in bboxes]
        middles = np.array([(2 * b[1] + height) / 2 for b, height in zip(bboxes, heights)])

        eps = np.median(heights) / 2
        dbscan = DBSCAN(eps=eps, min_samples=1)
        preds = dbscan.fit_predict(middles.reshape(-1, 1))

        srt = [dict(bbox=[], avgY=0) for _ in range(max(preds) + 1)]
        for i, pred in enumerate(preds):
            bbox = bboxes[i]
            cluster = preds[i]
            srt[cluster]['bbox'].append(bbox)
            srt[cluster]['avgY'] += bbox[1]

        for el in srt:
            el['avgY'] /= len(el['bbox'])
        srt = sorted(srt, key=lambda x: x['avgY'])
        srt = [el['bbox'] for el in srt]
        srt = [sorted(line, key=lambda x: x[0]) for line in srt]
        return srt
