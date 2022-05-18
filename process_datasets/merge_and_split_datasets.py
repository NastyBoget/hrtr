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

"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from process_datasets.process_rus import process_rus
from process_datasets.process_rus_kz import process_rus_kz
from process_datasets.process_synthetic import process_synthetic

datasets_handlers = dict(
    rus=process_rus,
    rus_kz=process_rus_kz,
    synthetic=process_synthetic
)


def merge_datasets(data_dir: str, img_dir: str, out_dir: str) -> None:
    """
    :param data_dir: full path to datasets directory
    :param img_dir: name of the directory with images
    :param out_dir: name of out directory e.g. merged
    :return:
    """
    for i, (dataset_dir, handler) in enumerate(datasets_handlers.items()):
        handler(data_dir=data_dir,
                out_dir=os.path.join(data_dir, out_dir),
                img_dir=img_dir,
                gt_file=f"gt{i}.txt")


def split_datasets(data_dir: str, out_dir: str, test_size: float = 0.3) -> None:
    # train = 0.7 test = 0.15 val = 0.15
    df_train = pd.DataFrame({"path": [], "word": []})
    df_val = pd.DataFrame({"path": [], "word": []})
    df_test = pd.DataFrame({"path": [], "word": []})

    merged_dir = os.path.join(data_dir, out_dir)
    for i in range(len(datasets_handlers)):
        df = pd.read_csv(os.path.join(merged_dir, f"gt{i}.txt"), sep="\t", names=["path", "word"])
        train, test_val = train_test_split(df, test_size=test_size)
        test, val = train_test_split(test_val, test_size=0.5)
        df_train = pd.concat([df_train, train], ignore_index=True)
        df_test = pd.concat([df_test, test], ignore_index=True)
        df_val = pd.concat([df_val, val], ignore_index=True)
        os.remove(os.path.join(merged_dir, f"gt{i}.txt"))

    df_train.to_csv(os.path.join(merged_dir, "gt_train.txt"), sep="\t", index=False, header=False)
    df_test.to_csv(os.path.join(merged_dir, "gt_test.txt"), sep="\t", index=False, header=False)
    df_val.to_csv(os.path.join(merged_dir, "gt_val.txt"), sep="\t", index=False, header=False)


if __name__ == "__main__":
    data_dir = "../datasets"
    out_dir = "merged"
    img_dir = "img"
    os.makedirs(os.path.join(data_dir, out_dir, img_dir), exist_ok=True)
    merge_datasets(data_dir, img_dir, out_dir)
    split_datasets(data_dir, out_dir)
