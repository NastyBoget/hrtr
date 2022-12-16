import os
import shutil
import zipfile

import pandas as pd
import wget


name_prefix = "synthetic"


def process_synthetic(data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
    """
    :param data_dir: directory path with dataset that includes img and ann directories
    :param out_dir: directory path for saving images and groundtruth file
    :param img_dir: directory name inside out_dir for saving images
    :param gt_file: name of the groundtruth file
    :return:
    """

    data_url = "https://at.ispras.ru/owncloud/index.php/s/5kZRtMH0Uis2PyW/download"
    root = os.path.join(data_dir, name_prefix)
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "archive.zip")
    print(f"Downloading {name_prefix} dataset...")
    wget.download(data_url, archive)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(root)
    data_dir = os.path.join(root, "synthetic")
    print("Dataset downloaded")

    df = pd.read_csv(os.path.join(data_dir, "gt.txt"), sep="\t", names=["path", "word"])
    df.to_csv(os.path.join(out_dir, f"train_{gt_file}"), sep="\t", index=False, header=False)
    print(f"{name_prefix} train dataset length: {df.shape[0]}")

    destination_img_dir = os.path.join(out_dir, img_dir)
    current_img_dir = os.path.join(data_dir, "img")
    for img_name in os.listdir(current_img_dir):
        shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, img_name))
    shutil.rmtree(root)


if __name__ == "__main__":
    data_dir = "../../datasets"
    out_dir = "synthetic"
    os.makedirs(os.path.join(data_dir, out_dir, "img"), exist_ok=True)
    process_synthetic(data_dir=data_dir,
                      out_dir=os.path.join(data_dir, out_dir),
                      img_dir="img",
                      gt_file=f"gt.txt")
