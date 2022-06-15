import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet

from lines_segmentation.binarization import binarize


class Binarization(object):
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray) -> Image:
        """
        :param img: np.array Image
        :return: binarized image (PIL)
        """
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = binarize(img)
        return Image.fromarray(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class TextClassifier:
    _nets = {}

    def __init__(self, on_gpu: bool, checkpoint_path: str, *, config: dict) -> None:
        self.logger = config.get("logger", logging.getLogger())
        self.classes = ['handwritten', 'typewritten']
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self._set_device(on_gpu)
        self._set_transform_image()

    @property
    def net(self) -> ResNet:
        # lazy loading and net sharing, comrade
        if self.checkpoint_path not in self._nets:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.location))
            model.to(self.device)
            self._nets[self.checkpoint_path] = model

        return self._nets[self.checkpoint_path]

    def _set_transform_image(self) -> None:
        self.transform = transforms.Compose([
            Binarization(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _set_device(self, on_gpu: bool) -> None:
        """
       Set device configuration
       """
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.location = lambda storage, loc: storage.cuda()
            self.logger.info("GPU is used")
        else:
            self.device = torch.device("cpu")
            self.location = 'cpu'
            self.logger.info("CPU is used")

    def predict(self, image: np.ndarray) -> str:
        self.net.eval()

        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).float().to(self.device)
            outputs = self.net(tensor_image)

        _, class_predicted = torch.max(outputs, 1)
        return self.classes[int(class_predicted[0])]
