import os
import shutil
import zipfile

import pandas as pd
import wget
from sklearn.model_selection import train_test_split

name_prefix = "rus"

# rus dataset https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset


def read_file(path: str) -> pd.DataFrame:
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


def process_rus(data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
    """
    :param data_dir: directory path with dataset that includes img and ann directories
    :param out_dir: directory path for saving images and groundtruth file
    :param img_dir: directory name inside out_dir for saving images
    :param gt_file: name of the groundtruth file
    :return:
    """

    data_url = "https://at.ispras.ru/owncloud/index.php/s/F6QwV1CY5ExbbbH/download"
    root = os.path.join(data_dir, name_prefix)
    os.makedirs(root)
    archive = os.path.join(root, "archive.zip")
    print(f"\nDownloading {name_prefix} dataset...")
    wget.download(data_url, archive)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(root)
    data_dir = root
    print("\nDataset downloaded")

    train_df = read_file(os.path.join(data_dir, "train.tsv"))
    test_df = read_file(os.path.join(data_dir, "test.tsv"))

    char_set = set()
    for _, row in train_df.iterrows():
        char_set = char_set | set(row["word"])

    for _, row in test_df.iterrows():
        char_set = char_set | set(row["word"])
    print(f"Rus char set: {repr(''.join(sorted(list(char_set))))}")

    test_df["path"] = f"{img_dir}/" + name_prefix + "test" + test_df.path
    train_df["path"] = f"{img_dir}/" + name_prefix + "train" + train_df.path

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    test_df.to_csv(os.path.join(out_dir, f"test_{gt_file}"), sep="\t", index=False, header=False)
    val_df.to_csv(os.path.join(out_dir, f"val_{gt_file}"), sep="\t", index=False, header=False)
    train_df.to_csv(os.path.join(out_dir, f"train_{gt_file}"), sep="\t", index=False, header=False)
    print(f"{name_prefix}: train dataset length: {train_df.shape[0]}")
    print(f"{name_prefix}: val dataset length: {val_df.shape[0]}")
    print(f"{name_prefix}: test dataset length: {test_df.shape[0]}")

    for img_dir_name in ("train", "test"):
        current_img_dir = os.path.join(data_dir, img_dir_name)
        destination_img_dir = os.path.join(out_dir, img_dir)
        for img_name in os.listdir(current_img_dir):
            new_img_name = name_prefix + img_dir_name + img_name
            shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
    shutil.rmtree(root)


if __name__ == "__main__":
    data_dir = "../../datasets"
    out_dir = "rus_out"
    os.makedirs(os.path.join(data_dir, out_dir, "img"), exist_ok=True)
    process_rus(data_dir=data_dir,
                out_dir=os.path.join(data_dir, out_dir),
                img_dir="img",
                gt_file=f"gt.txt")
