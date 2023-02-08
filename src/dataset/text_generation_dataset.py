from typing import Any, Tuple

from PIL import Image
from torch.utils.data import Dataset

from src.dataset.augmentation.text_augmentation import ImageGenerator


class TextGenerationDataset(Dataset):

    def __init__(self, opt: Any) -> None:
        self.n_samples = opt.generate_num
        self.image_generator = ImageGenerator(fonts_dir=opt.fonts_dir)
        self.rgb = opt.rgb

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        return self.image_generator.get_image(rgb=self.rgb)
