import logging
import os
import shutil
import tempfile
import zipfile

import pandas as pd
import wget
from sklearn.model_selection import train_test_split

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor


class CyrillicDatasetProcessor(AbstractDatasetProcessor):
    """Process cyrillic dataset https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset """

    __dataset_name = "cyrillic"
    __charset = ' !"%\'()+,-./0123456789:;=?R[]bcehioprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№'

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.data_url = "https://at.ispras.ru/owncloud/index.php/s/qruHwU5iSOMUrw0/download"
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
            data_dir = root
            self.logger.info("Dataset downloaded")

            train_df = self.__read_file(os.path.join(data_dir, "train.tsv"))
            test_df = self.__read_file(os.path.join(data_dir, "test.tsv"))

            char_set = set()
            for _, row in train_df.iterrows():
                char_set = char_set | set(row["word"])

            for _, row in test_df.iterrows():
                char_set = char_set | set(row["word"])
            self.__charset = char_set
            self.logger.info(f"{self.dataset_name} char set: {repr(''.join(sorted(list(char_set))))}")

            test_df["path"] = f"{img_dir}/{self.dataset_name}_test_" + test_df.path
            train_df["path"] = f"{img_dir}/{self.dataset_name}_train_" + train_df.path
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

            test_df.to_csv(os.path.join(out_dir, f"test_{gt_file}"), sep="\t", index=False, header=False)
            val_df.to_csv(os.path.join(out_dir, f"val_{gt_file}"), sep="\t", index=False, header=False)
            train_df.to_csv(os.path.join(out_dir, f"train_{gt_file}"), sep="\t", index=False, header=False)
            self.logger.info(f"{self.dataset_name} length: train = {train_df.shape[0]}; val = {val_df.shape[0]}; test = {test_df.shape[0]}")

            for img_dir_name in ("train", "test"):
                current_img_dir = os.path.join(data_dir, img_dir_name)
                destination_img_dir = os.path.join(out_dir, img_dir)
                for img_name in os.listdir(current_img_dir):
                    new_img_name = f"{self.dataset_name}_{img_dir_name}_{img_name}"
                    shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
            shutil.rmtree(root)

    def __read_file(self, path: str) -> pd.DataFrame:
        result = {"path": [], "word": []}
        with open(path, 'r') as data:
            datalist = data.readlines()

        for line in datalist:
            image_path, label = line.strip('\n').split('.png')
            label = label.strip()
            label.replace("\r", "")
            label.replace("\t", " ")
            result["path"].append(f"{image_path}.png")
            result["word"].append(label)
        return pd.DataFrame(result)
