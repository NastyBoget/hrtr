import logging
from typing import Any, Tuple

from PIL import Image
from torch.utils.data import Dataset

from src.dataset.augmentation.text_augmentation import ImageGenerator


class TextGenerationDataset(Dataset):

    def __init__(self, opt: Any, logger: logging.Logger) -> None:
        self.n_samples = opt.generate_num
        self.image_generator = ImageGenerator(fonts_dir=opt.fonts_dir, rgb=opt.rgb, background_color=255, text_color=0)
        self.logger = logger

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        img, label = None, None
        while img is None:
            try:
                img, label = self.image_generator.get_image()
            except Exception as e:
                self.logger.info(f"Got exception in TextGeneration: {e}")
                img = None
        return img, label
