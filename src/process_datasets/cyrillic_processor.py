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
    __charset = ' !"%\'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№'

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

            test_df["path"] = f"{img_dir}/{self.dataset_name}_test_" + test_df.path
            train_df["path"] = f"{img_dir}/{self.dataset_name}_train_" + train_df.path
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
            test_df["stage"] = "test"
            val_df["stage"] = "val"
            train_df["stage"] = "train"
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            df["sample_id"] = df.index
            df.to_csv(os.path.join(out_dir, gt_file), sep=",", index=False)

            char_set = set()
            for _, row in df.iterrows():
                char_set = char_set | set(row["text"])
            self.__charset = char_set
            self.logger.info(f"{self.dataset_name} char set: {repr(''.join(sorted(list(char_set))))}")
            self.logger.info(f"{self.dataset_name} dataset length: train = {len(df[df.stage == 'train'])}; "
                             f"val = {len(df[df.stage == 'val'])}; "
                             f"test = {len(df[df.stage == 'test'])}")

            for img_dir_name in ("train", "test"):
                current_img_dir = os.path.join(data_dir, img_dir_name)
                destination_img_dir = os.path.join(out_dir, img_dir)
                for img_name in os.listdir(current_img_dir):
                    new_img_name = f"{self.dataset_name}_{img_dir_name}_{img_name}"
                    shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
            shutil.rmtree(root)

    def __read_file(self, path: str) -> pd.DataFrame:
        result = {"path": [], "text": []}
        with open(path, 'r') as data:
            datalist = data.readlines()

        for line in datalist:
            image_path, label = line.strip('\n').split('.png')
            label = label.strip()
            label.replace("\r", "")
            label.replace("\t", " ")
            result["path"].append(f"{image_path}.png")
            result["text"].append(label)
        return pd.DataFrame(result)
