import logging
import os
import shutil
import tempfile
import zipfile

import pandas as pd
import wget
from sklearn.model_selection import train_test_split

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor


class SyntheticDatasetProcessor(AbstractDatasetProcessor):

    __dataset_names = ["synthetic_hkr", "gan_hkr", "stackmix_hkr", "synthetic_cyrillic", "gan_cyrillic", "stackmix_cyrillic"]

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()

        self.data_urls = {
            "synthetic_hkr": "",
            "gan_hkr": "",
            "stackmix_hkr": "",  # 2476836 images
            "synthetic_cyrillic": "",
            "gan_cyrillic": "",
            "stackmix_cyrillic": ""  # 3700269 images
        }
        self.logger = logger

    def can_process(self, d_name: str) -> bool:
        return d_name in self.__dataset_names

    @property
    def charset(self) -> str:
        return self.__charset

    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str, dataset_name: str) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            archive = os.path.join(data_dir, "archive.zip")
            self.logger.info(f"Downloading {dataset_name} dataset...")
            wget.download(self.data_urls[dataset_name], archive)
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            data_dir = os.path.join(data_dir, dataset_name)
            self.logger.info("Dataset downloaded")

            df = pd.read_csv(os.path.join(data_dir, "gt.txt"), sep="\t", names=["path", "text"])
            df["path"] = f"{img_dir}" + df.path.str[3:]  # data startswith img
            char_set = set()
            for _, row in df.iterrows():
                char_set = char_set | set(row["text"])
            self.logger.info(f"{dataset_name} char set: {repr(''.join(sorted(list(char_set))))}")
            self.__charset = char_set

            train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
            train_df["stage"] = "train"
            val_df["stage"] = "val"
            df = pd.concat([train_df, val_df], ignore_index=True)

            self.logger.info(f"{dataset_name} dataset length: train = {train_df.shape[0]}; val = {val_df.shape[0]}")
            df["sample_id"] = df.index
            df.to_csv(os.path.join(out_dir, gt_file), sep=",", index=False)

            destination_img_dir = os.path.join(out_dir, img_dir)
            current_img_dir = os.path.join(data_dir, "img")
            for img_name in os.listdir(current_img_dir):
                shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, img_name))
