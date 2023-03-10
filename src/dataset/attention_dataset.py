import os
from typing import Any, Tuple

import cv2
import pandas as pd
from PIL import Image

from dataset.abstract_dataset import AbstractDataset


class AttentionDataset(AbstractDataset):

    def __init__(self, df: pd.DataFrame, data_dir: str, opt: Any, transforms=None):
        super().__init__(df, data_dir)
        self.rgb = opt.rgb
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        assert idx <= len(self), 'index range error'
        img = cv2.imread(os.path.join(self.data_dir, self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        img = Image.fromarray(img).convert('RGB') if self.rgb else Image.fromarray(img).convert('L')

        text = str(self.texts[idx])
        return img, text
