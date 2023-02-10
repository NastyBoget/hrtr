import os
import random
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageFont, ImageDraw

from src.dataset.augmentation.text_generator import TextGenerator

font2not_allowed_symbols = {
    "Abram.ttf": "Ъ",
    "Benvolio.ttf": "Ъ",
    "Capuletty.ttf": "Ъ",
    "Eskal.ttf": "Ъ",
    "Gregory.ttf": "Ъ",
    "Gogol.ttf": "\"%()?",
    "Lorenco.ttf": "Ъ",
    "Marutya.ttf": "ЬЫЪ",
    "Merkucio.ttf": "Ъ",
    "Montekky.ttf": "Ъ",
    "Pushkin.ttf": "%",
    "Tesla.otf": "ЙЩЬЫЪЭЮЯйщьыъэюя",
    "Tibalt.ttf": "Ъ",
    "Voronov.ttf": "ЬЫЪ"
}


class ImageGenerator:

    def __init__(self,
                 fonts_dir: str,
                 rgb: bool,
                 background_color: Union[tuple, int] = (255, 255, 255),
                 text_color: Union[tuple, int] = (0, 0, 0)) -> None:
        self.rgb = rgb
        if rgb and isinstance(background_color, int):
            self.background_color = (background_color, background_color, background_color)
        elif not rgb and isinstance(background_color, tuple):
            self.background_color = background_color[0]
        else:
            self.background_color = background_color
        self.text_color = text_color[0] if not rgb and isinstance(text_color, tuple) else text_color

        self.fonts_dir = fonts_dir
        self.font_names = sorted([font_name for font_name in os.listdir(fonts_dir) if font_name.lower().endswith((".ttf", ".otf"))])
        self.text_generator = TextGenerator()

    def get_image(self) -> Tuple[Image.Image, str]:
        font_size = random.randint(30, 100)
        x_margin, y_margin = random.randint(0, 10), random.randint(0, 10)
        font_name = self.font_names[random.randint(0, len(self.font_names) - 1)]
        font = ImageFont.truetype(os.path.join(self.fonts_dir, font_name), font_size)

        text = ""
        while len(text) == 0:
            text = self.text_generator.get_text()
            if font_name in font2not_allowed_symbols:
                for sym in font2not_allowed_symbols[font_name]:
                    text = text.replace(sym, "")

        img_size = (font_size * 10, font_size * len(text) * 10, 3) if self.rgb else (font_size * 10, font_size * len(text) * 10)
        img = np.zeros(img_size, dtype=np.uint8)
        img[:, :] = self.background_color
        img = Image.fromarray(img) if self.rgb else Image.fromarray(img, 'L')
        draw = ImageDraw.Draw(img)

        x, y = font_size, font_size
        x_min, y_min, x_max, y_max = draw.textbbox((x, y), text, font)
        draw.text((x, y), text, self.text_color, font=font)

        return img.crop((x_min - x_margin, y_min - y_margin, x_max + x_margin, y_max + y_margin)), text
