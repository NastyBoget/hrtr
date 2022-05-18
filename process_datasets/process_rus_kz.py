import json
import os
import shutil
import zipfile

import pandas as pd
import wget

from utils import check_valid_label


name_prefix = "rus_kz"

# rus_kz dataset https://drive.google.com/drive/folders/1zOAOD_E7FWW9NrRAXSci0zmk30yqJS4o


def process_rus_kz(data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
    """
    :param data_dir: directory path with dataset that includes img and ann directories
    :param out_dir: directory path for saving images and groundtruth file
    :param img_dir: directory name inside out_dir for saving images
    :param gt_file: name of the groundtruth file
    :return:
    """
    data_url = "https://at.ispras.ru/owncloud/index.php/s/qeY5idmhbKOipxf/download"
    root = os.path.join(data_dir, name_prefix)
    os.makedirs(root)
    archive = os.path.join(root, "archive.zip")
    print(f"Downloading {name_prefix} dataset...")
    wget.download(data_url, archive)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(root)
    data_dir = os.path.join(root, "HK_dataset")
    print("Dataset downloaded")

    ann_dir = os.path.join(data_dir, "ann")
    data_dict = {"path": [], "word": []}

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".json"):
            continue

        with open(os.path.join(ann_dir, ann_file)) as f:
            ann_f = json.load(f)
        data_dict["path"].append(ann_f['name'])
        data_dict["word"].append(ann_f["description"])

    data_df = pd.DataFrame(data_dict)
    data_df["path"] = f"{img_dir}/" + name_prefix + data_df.path
    data_df["valid"] = data_df.apply(lambda row: check_valid_label(row[1]), axis=1)
    data_df = data_df[data_df.valid].drop(["valid"], axis=1)
    print(f"{name_prefix} dataset length: {data_df.shape[0]}")

    data_df.to_csv(os.path.join(out_dir, gt_file), sep="\t", index=False, header=False)

    current_img_dir = os.path.join(data_dir, "img")
    destination_img_dir = os.path.join(out_dir, img_dir)
    for img_name in os.listdir(current_img_dir):
        new_img_name = name_prefix + img_name
        shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
    shutil.rmtree(root)


if __name__ == "__main__":
    data_dir = "../datasets"
    out_dir = "rus_kz_out"
    os.makedirs(os.path.join(data_dir, out_dir, "img"), exist_ok=True)
    process_rus_kz(data_dir=data_dir,
                   out_dir=os.path.join(data_dir, out_dir),
                   img_dir="img",
                   gt_file=f"gt.txt")
