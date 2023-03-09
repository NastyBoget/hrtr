import json
import logging
import os
import shutil
import tempfile
import zipfile

import pandas as pd
import wget

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor


class HKRDatasetProcessor(AbstractDatasetProcessor):
    """Process HKR dataset https://github.com/abdoelsayed2016/HKR_Dataset splitting: https://github.com/bosskairat/Dataset """

    __dataset_name = "hkr"
    __charset = ' !(),-.:;?HoАБВГДЕЖЗИЙКЛМНОПРСТУФХЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёғҚқҮӨө–—…'

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()

        self.data_url = "https://at.ispras.ru/owncloud/index.php/s/llLrs5lORQQXCYt/download"
        self.logger = logger

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def charset(self) -> str:
        return self.__charset

    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            root = os.path.join(data_dir, self.dataset_name)
            os.makedirs(root)
            archive = os.path.join(root, "archive.zip")
            self.logger.info(f"Downloading {self.dataset_name} dataset...")
            wget.download(self.data_url, archive)
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(root)
            data_dir = os.path.join(root, "HKR_dataset")
            self.logger.info("Dataset downloaded")

            data_df = self.__get_split(data_dir)
            char_set = set()
            for _, row in data_df.iterrows():
                char_set = char_set | set(row["text"])
            self.logger.info(f"HKR char set: {repr(''.join(sorted(list(char_set))))}")
            self.__charset = char_set

            data_df["path"] = f"{img_dir}/{self.dataset_name}_" + data_df.path
            data_df["sample_id"] = data_df.index
            data_df.to_csv(os.path.join(out_dir, gt_file), sep=",", index=False)

            self.logger.info(f"{self.dataset_name} dataset length: train = {len(data_df[data_df.stage == 'train'])}; "
                             f"val = {len(data_df[data_df.stage == 'val'])}; "
                             f"test1 = {len(data_df[data_df.stage == 'test1'])}; "
                             f"test2 = {len(data_df[data_df.stage == 'test2'])}")

            current_img_dir = os.path.join(data_dir, "img")
            destination_img_dir = os.path.join(out_dir, img_dir)
            for img_name in os.listdir(current_img_dir):
                new_img_name = f"{self.dataset_name}_{img_name}"
                shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
            shutil.rmtree(root)

    def __get_split(self, data_dir: str) -> pd.DataFrame:
        name2stage = {}
        split_df = pd.read_csv(os.path.join(data_dir, "HKR_splitting.csv"))
        for _, row in split_df.iterrows():
            name2stage[row['id']] = row['stage']
        ann_dir = os.path.join(data_dir, "ann")
        data_dict = {"path": [], "text": [], "stage": []}

        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith(".json"):
                continue

            with open(os.path.join(ann_dir, ann_file)) as f:
                ann_f = json.load(f)
            data_dict["stage"].append(name2stage[ann_f['name']])
            data_dict["path"].append(f"{ann_f['name']}.jpg")
            data_dict["text"].append(ann_f["description"])
        data_df = pd.DataFrame(data_dict)
        return data_df
