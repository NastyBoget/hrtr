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

    __dataset_name = "synthetic"
    __charset = ' !"%(),-.0123456789:;?АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя'

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        # small (60K) https://at.ispras.ru/owncloud/index.php/s/jgApabH3GK2bgUG/download
        # large (4M) https://at.ispras.ru/owncloud/index.php/s/d8TDv92ayoGvFiM/download
        self.data_url = "https://at.ispras.ru/owncloud/index.php/s/jgApabH3GK2bgUG/download"
        self.logger = logger

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def charset(self) -> str:
        return self.__charset

    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            archive = os.path.join(data_dir, "archive.zip")
            self.logger.info(f"Downloading {self.dataset_name} dataset...")
            wget.download(self.data_url, archive)
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            data_dir = os.path.join(data_dir, "synthetic")
            self.logger.info("Dataset downloaded")

            df = pd.read_csv(os.path.join(data_dir, "gt.txt"), sep="\t", names=["path", "text"])
            df["path"] = f"{img_dir}" + df.path.str[3:]  # data startswith img
            char_set = set()
            for _, row in df.iterrows():
                char_set = char_set | set(row["text"])
            self.logger.info(f"{self.dataset_name} char set: {repr(''.join(sorted(list(char_set))))}")
            self.__charset = char_set

            train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
            train_df["stage"] = "train"
            val_df["stage"] = "val"
            df = pd.concat([train_df, val_df], ignore_index=True)

            self.logger.info(f"{self.dataset_name} dataset length: train = {train_df.shape[0]}; val = {val_df.shape[0]}")
            df["sample_id"] = df.index
            df.to_csv(os.path.join(out_dir, gt_file), sep=",", index=False)

            destination_img_dir = os.path.join(out_dir, img_dir)
            current_img_dir = os.path.join(data_dir, "img")
            for img_name in os.listdir(current_img_dir):
                shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, img_name))
