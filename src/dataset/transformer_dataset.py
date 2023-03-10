import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skimage import color

from src.dataset.abstract_dataset import AbstractDataset
from src.transformer_model.custom_functions import SmartResize


class TransformerDataset(AbstractDataset):

    def __init__(self, df: pd.DataFrame, data_dir: str, tokenizers: dict, transforms=None):
        super().__init__(df, data_dir)
        self.tokenizers = tokenizers
        self.transforms = transforms
        self.resize = SmartResize(384, 96, stretch=(1.0, 1.0), fillcolor=255)
        self.normalize = A.Normalize()

    def __getitem__(self, idx: int) -> dict:
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.data_dir, self.paths[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text = str(self.texts[idx])

        img, scale_coeff = self.resize(img)
        if self.transforms:
            img = self.transforms(image=img)['image']

        image_mask = [x.mean() >= 0.999 for x in np.split(color.rgb2gray(img) / 255, np.arange(384 // 24, 384, 384 // 24), axis=1)]
        mask_false_count = len(image_mask) - image_mask[::-1].index(False)
        image_mask = [False] * mask_false_count + [True] * (len(image_mask) - mask_false_count)

        # Normalize
        img = self.normalize(image=img.astype(np.float32))["image"]
        # To Grayscale
        img = color.rgb2gray(img)
        # To Tensor
        img = torch.from_numpy(img).unsqueeze(0)

        out_dict = {'image': img, 'image_mask': image_mask, 'text': text, 'img_path': img_path, 'scale_coeff': scale_coeff}
        for tokenizer_name, tokenizer in self.tokenizers.items():
            out_dict['enc_text_' + tokenizer_name] = torch.LongTensor(tokenizer.encode([text])[0])
        return out_dict
