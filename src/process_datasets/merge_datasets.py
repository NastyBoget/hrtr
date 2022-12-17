"""
initial structure       result structure
================================================

datasets                datasets
|                       |
|----rus                |--merged
|    |                      |
|    |---test               |---img
|    |                      |
|    |---train              |---gt_train.txt
|    |                      |
|    |---test.csv           |---gt_val.txt
|    |                      |
|    |---train.csv          |---gt_test.txt
|
|----rus_kz
     |
     |---ann
     |
     |---img
     |
     |---HKR_splitting.csv

"""
import os
from typing import List

import pandas as pd

from src.process_datasets.process_rus import process_rus
from src.process_datasets.process_rus_kz import process_rus_kz
from src.process_datasets.process_synthetic import process_synthetic

datasets_handlers = dict(
    rus=process_rus,
    hkr=process_rus_kz,
    synthetic=process_synthetic
)


def merge_datasets(data_dir: str, img_dir: str, out_dir: str, datasets_list: List[str]) -> None:
    """
    :param data_dir: full path to datasets directory
    :param img_dir: name of the directory with images
    :param out_dir: name of out directory e.g. merged
    :return:
    """
    key2df = dict()

    for i, dataset_name in enumerate(datasets_list):
        datasets_handlers[dataset_name](data_dir=data_dir,
                                        out_dir=os.path.join(data_dir, out_dir),
                                        img_dir=img_dir,
                                        gt_file=f"gt{i}.txt")
        for gt_file in os.listdir(os.path.join(data_dir, out_dir)):
            if not gt_file.endswith(f"{i}.txt"):
                continue

            key, _ = gt_file.split("_")
            if key not in key2df:
                key2df[key] = pd.DataFrame({"path": [], "word": []})

            df = pd.read_csv(os.path.join(data_dir, out_dir, f"{key}_gt{i}.txt"), sep="\t", names=["path", "word"])
            key2df[key] = pd.concat([key2df[key], df], ignore_index=True)

    for key in key2df:
        key2df[key].to_csv(os.path.join(data_dir, out_dir, f"gt_{key}.txt"), sep="\t", index=False, header=False)


if __name__ == "__main__":
    data_dir = "../../datasets"
    out_dir = "merged"
    img_dir = "img"
    datasets = ["rus", "synthetic"]
    os.makedirs(os.path.join(data_dir, out_dir, img_dir), exist_ok=True)
    merge_datasets(data_dir, img_dir, out_dir, datasets)
