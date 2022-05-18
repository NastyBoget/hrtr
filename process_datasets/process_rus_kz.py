import json
import os
import shutil

import pandas as pd


def process_rus_kz(data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
    """
    :param data_dir: directory path with dataset that includes img and ann directories
    :param out_dir: directory path for saving images and groundtruth file
    :param img_dir: directory name inside out_dir for saving images
    :param gt_file: name of the groundtruth file
    :return:
    """
    # data_dir = "../datasets/rus_kz"
    ann_dir = os.path.join(data_dir, "ann")
    data_dict = {"path": [], "word": []}
    name_prefix = "rus_kz"

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".json"):
            continue

        with open(os.path.join(ann_dir, ann_file)) as f:
            ann_f = json.load(f)
        data_dict["path"].append(ann_f['name'])
        data_dict["word"].append(ann_f["description"])

    data_df = pd.DataFrame(data_dict)
    data_df["path"] = f"{img_dir}/" + name_prefix + data_df.path
    data_df.to_csv(os.path.join(out_dir, gt_file), sep="\t", index=False, header=False)

    current_img_dir = os.path.join(data_dir, "img")
    destination_img_dir = os.path.join(out_dir, img_dir)
    for img_name in os.listdir(current_img_dir):
        new_img_name = name_prefix + img_name
        shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
