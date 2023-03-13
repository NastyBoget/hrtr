import os
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch

from dataset.abstract_dataset import AbstractDataset
from src.ctc_model.utils import resize_if_need, make_img_padding


class CTCDataset(AbstractDataset):

    def __init__(self, df_list: List[pd.DataFrame], data_dir: str, ctc_labeling, transforms=None):
        super().__init__(df_list, data_dir)
        self.ctc_labeling = ctc_labeling
        self.transforms = transforms
        self.image_w = 2048
        self.image_h = 128

    def __getitem__(self, idx: int) -> dict:
        assert idx <= len(self), 'index range error'

        df_idx = idx % len(self.texts)
        current_idx = self.current_idx_list[df_idx]
        self.current_idx_list[df_idx] = (self.current_idx_list[df_idx] + 1) % len(self.texts[df_idx])

        img = cv2.imread(os.path.join(self.data_dir, self.paths[df_idx][current_idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, coef = self.resize_image(img)

        if self.transforms:
            img = self.transforms(image=img)['image']

        text = str(self.texts[df_idx][current_idx])
        encoded = self.ctc_labeling.encode(text)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return {'id': idx, 'image': img, 'gt_text': text, 'coef': coef, 'encoded': torch.tensor(encoded, dtype=torch.int32)}

    def resize_image(self, image):
        image, coef = resize_if_need(image, self.image_h, self.image_w)
        image = make_img_padding(image, self.image_h, self.image_w)
        return image, coef
