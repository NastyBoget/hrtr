import math
from typing import Tuple, Any, Iterable

import torch
from PIL import Image
from torchvision import transforms as transforms


class NormalizePAD(object):

    def __init__(self, max_size: Tuple[int, int, int], pad_type: str = 'right') -> None:
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.pad_type = pad_type

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img  # right PAD
        if self.max_size[2] != w:  # add border PAD
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return pad_img


class ResizeNormalize(object):

    def __init__(self, size: Tuple[int, int], interpolation: Any = Image.BICUBIC) -> None:
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class AlignCollate(object):

    def __init__(self, img_h: int = 32, img_w: int = 100, keep_ratio_with_pad: bool = False) -> None:
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch: Iterable) -> Tuple[torch.Tensor, Any]:
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:
            resized_max_w = self.img_w
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.img_h, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.img_h * ratio) > self.img_w:
                    resized_w = self.img_w
                else:
                    resized_w = math.ceil(self.img_h * ratio)

                resized_image = image.resize((resized_w, self.img_h), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.img_w, self.img_h))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels
