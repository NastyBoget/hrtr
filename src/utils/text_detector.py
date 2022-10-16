import logging
import os
from typing import Optional, List

import numpy as np
import torch
from doctr.models import detection_predictor
from doctr.models.detection.predictor import DetectionPredictor


class TextDetector:

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 with_vertical_text_detection: bool = False,
                 custom_model: bool = False,
                 arch: Optional[str] = None) -> None:
        self.logger = logging.getLogger()
        self._net = None
        self.checkpoint_path = checkpoint_path if checkpoint_path is None else os.path.abspath(checkpoint_path)
        self.with_vertical_text_detection = with_vertical_text_detection
        self.custom_model = custom_model
        self.arch = arch
        assert(not self.custom_model or arch is not None)

    @property
    def net(self) -> DetectionPredictor:
        """
    (x_top_left, y_top_left)
        |----------------------------------|
        |                                  |
        |----------------------------------|
                                    (x_bottom_right, y_bottom_right)

        Predict consists from List of bounding box with a format
        (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
        :return: Text Detection model
        """
        # lazy loading net
        if not self._net:
            if self.custom_model:
                self._net = detection_predictor(arch=self.arch, pretrained=False).eval()
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                self._net.model.load_state_dict(checkpoint)
                self._net.eval()
            elif self.with_vertical_text_detection:
                self._net = detection_predictor(arch='db_resnet50_rotation', pretrained=True).eval()
            else:
                self._net = detection_predictor(arch='db_resnet50', pretrained=True).eval()

        return self._net

    def predict(self, image: np.ndarray) -> List[tuple]:
        """
        :param image: input batch of image with some text
        :return: prediction: List[tuple] - text coords prediction, list of bboxes
        (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
        """
        h, w, _ = image.shape
        preds = self.net([image])
        boxes = [(int(pred[0] * w), int(pred[1] * h), int(pred[2] * w), int(pred[3] * h)) for pred in preds[0]]
        return boxes
