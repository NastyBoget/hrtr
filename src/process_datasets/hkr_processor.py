import json
import logging
import os
import shutil
import sys
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

        # binarized data TODO
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
                char_set = char_set | set(row["word"])
            self.logger.info(f"HKR char set: {repr(''.join(sorted(list(char_set))))}")
            self.__charset = char_set

            data_df["path"] = f"{self.dataset_name}_" + data_df.path
            train_df = data_df[data_df.stage == "train"]
            train_df = train_df.drop(columns=['stage'])
            val_df = data_df[data_df.stage == "val"]
            val_df = val_df.drop(columns=['stage'])
            test1_df = data_df[data_df.stage == "test1"]
            test1_df = test1_df.drop(columns=['stage'])
            test2_df = data_df[data_df.stage == "test2"]
            test2_df = test2_df.drop(columns=['stage'])

            test1_df.to_csv(os.path.join(out_dir, f"test1_{gt_file}"), sep="\t", index=False, header=False)
            test2_df.to_csv(os.path.join(out_dir, f"test2_{gt_file}"), sep="\t", index=False, header=False)
            val_df.to_csv(os.path.join(out_dir, f"val_{gt_file}"), sep="\t", index=False, header=False)
            train_df.to_csv(os.path.join(out_dir, f"train_{gt_file}"), sep="\t", index=False, header=False)
            self.logger.info(f"{self.dataset_name} dataset length: train = {train_df.shape[0]}; val = {val_df.shape[0]}; test1 = {test1_df.shape[0]}; test2 = {test2_df.shape[0]}")  # noqa

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
        data_dict = {"path": [], "word": [], "stage": []}

        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith(".json"):
                continue

            with open(os.path.join(ann_dir, ann_file)) as f:
                ann_f = json.load(f)
            data_dict["stage"].append(name2stage[ann_f['name']])
            data_dict["path"].append(f"{ann_f['name']}.jpg")
            data_dict["word"].append(ann_f["description"])
        data_df = pd.DataFrame(data_dict)
        return data_df


if __name__ == "__main__":
    out_path = "/Users/anastasiabogatenkova/work/hrtr/transformer_model"
    img_dir = "img"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    p = HKRDatasetProcessor(logger=root)
    p.process_dataset(out_path, img_dir, "hkr.txt")
