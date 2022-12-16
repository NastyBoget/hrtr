import os
import shutil
import zipfile

import pandas as pd
import wget


name_prefix = "rus"

# rus dataset https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset


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
    print(f"Downloading {name_prefix} dataset...")
    wget.download(data_url, archive)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(root)
    data_dir = root
    print("Dataset downloaded")

    train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t", names=["path", "word"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t", names=["path", "word"])

    test_df["path"] = f"{img_dir}/" + name_prefix + "test" + test_df.path
    train_df["path"] = f"{img_dir}/" + name_prefix + "train" + train_df.path

    test_df.to_csv(os.path.join(out_dir, f"test_{gt_file}"), sep="\t", index=False, header=False)
    test_df.to_csv(os.path.join(out_dir, f"train_{gt_file}"), sep="\t", index=False, header=False)
    print(f"{name_prefix}: train dataset length: {train_df.shape[0]}")
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
