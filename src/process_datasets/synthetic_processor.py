import logging
import os
import shutil

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor


class SyntheticDatasetProcessor(AbstractDatasetProcessor):

    __dataset_names = ["synthetic_hkr", "gan_hkr", "stackmix_hkr", "synthetic_cyrillic", "gan_cyrillic", "stackmix_cyrillic"]

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.logger = logger

    def can_process(self, d_name: str) -> bool:
        return d_name in self.__dataset_names

    @property
    def charset(self) -> str:
        return self.__charset

    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str, dataset_name: str) -> None:
        tmp_dir = os.path.join(out_dir, "tmp")
        os.makedirs(tmp_dir)

        self.logger.info(f"Downloading {dataset_name} dataset...")
        dataset = datasets.load_dataset(f"nastyboget/{dataset_name}", cache_dir=tmp_dir)
        self.logger.info("Dataset downloaded")

        df = pd.DataFrame({"text": dataset["train"]["text"], "path": dataset["train"]["name"]})
        df["path"] = f"{img_dir}/" + df.path

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
        for img_path in dataset["train"]["path"]:
            shutil.move(img_path, os.path.join(destination_img_dir, os.path.basename(img_path)))
        shutil.rmtree(tmp_dir)
