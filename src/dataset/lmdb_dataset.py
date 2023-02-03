import logging
import re
import sys
from typing import Any, Tuple

import lmdb
import six
from PIL import Image
from torch.utils.data import Dataset


class LmdbDataset(Dataset):

    def __init__(self, root: str, opt: Any, logger: logging.Logger) -> None:
        self.root = root
        self.opt = opt
        self.logger = logger
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            logger.error(f'Cannot create lmdb from {root}')
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get('num-samples'.encode()))

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.n_samples)]
            else:
                self.filtered_index_list = []
                for index in range(self.n_samples):
                    index += 1  # lmdb starts with 1
                    label_key = f'label-{index:09d}'.encode()
                    label = txn.get(label_key).decode('utf-8')
                    if len(label) > self.opt.batch_max_length:
                        continue

                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)
                self.n_samples = len(self.filtered_index_list)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = f'image-{index:09d}'.encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB') if self.opt.rgb else Image.open(buf).convert('L')
            except IOError:
                self.logger.info(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = Image.new('RGB', (self.opt.img_w, self.opt.img_h)) if self.opt.rgb else Image.new('L', (self.opt.img_w, self.opt.img_h))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return img, label
