import os
import shutil

import pandas as pd


def process_rus(data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
    """
    :param data_dir: directory path with dataset that includes img and ann directories
    :param out_dir: directory path for saving images and groundtruth file
    :param img_dir: directory name inside out_dir for saving images
    :param gt_file: name of the groundtruth file
    :return:
    """
    name_prefix = "rus"
    train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t", names=["path", "word"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t", names=["path", "word"])

    test_df["path"] = f"{img_dir}/" + name_prefix + "test" + test_df.path
    train_df["path"] = f"{img_dir}/" + name_prefix + "train" + train_df.path

    result_df = pd.concat([test_df, train_df], ignore_index=True)
    result_df.to_csv(os.path.join(out_dir, gt_file), sep="\t", index=False, header=False)

    for img_dir_name in ("train", "test"):
        current_img_dir = os.path.join(data_dir, img_dir_name)
        destination_img_dir = os.path.join(out_dir, img_dir)
        for img_name in os.listdir(current_img_dir):
            new_img_name = name_prefix + img_dir_name + img_name
            shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
