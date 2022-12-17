import json
import os
import shutil
import zipfile

import pandas as pd
import wget


name_prefix = "hkr"

# more data https://drive.google.com/drive/folders/1zOAOD_E7FWW9NrRAXSci0zmk30yqJS4o https://github.com/abdoelsayed2016/KOHTD
# rus_kz dataset https://github.com/abdoelsayed2016/HKR_Dataset splitting: https://github.com/bosskairat/Dataset


def process_rus_kz(data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
    """
    :param data_dir: directory path with dataset that includes img and ann directories
    :param out_dir: directory path for saving images and groundtruth file
    :param img_dir: directory name inside out_dir for saving images
    :param gt_file: name of the groundtruth file
    :return:
    """
    data_url = "https://at.ispras.ru/owncloud/index.php/s/llLrs5lORQQXCYt/download"
    root = os.path.join(data_dir, name_prefix)
    os.makedirs(root)
    archive = os.path.join(root, "archive.zip")
    print(f"\nDownloading {name_prefix} dataset...")
    wget.download(data_url, archive)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(root)
    data_dir = os.path.join(root, "HKR_dataset")
    print("\nDataset downloaded")

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
    char_set = set()
    for _, row in data_df.iterrows():
        char_set = char_set | set(row["word"])
    print(f"HKR char set: {repr(''.join(sorted(list(char_set))))}")

    data_df["path"] = f"{img_dir}/" + name_prefix + data_df.path
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
    print(f"{name_prefix}: train dataset length: {train_df.shape[0]}")
    print(f"{name_prefix}: val dataset length: {val_df.shape[0]}")
    print(f"{name_prefix}: test1 dataset length: {test1_df.shape[0]}")
    print(f"{name_prefix}: test2 dataset length: {test2_df.shape[0]}")

    current_img_dir = os.path.join(data_dir, "img")
    destination_img_dir = os.path.join(out_dir, img_dir)
    for img_name in os.listdir(current_img_dir):
        new_img_name = name_prefix + img_name
        shutil.move(os.path.join(current_img_dir, img_name), os.path.join(destination_img_dir, new_img_name))
    shutil.rmtree(root)


if __name__ == "__main__":
    data_dir = "../../datasets"
    out_dir = "hkr_out"
    os.makedirs(os.path.join(data_dir, out_dir, "img"), exist_ok=True)
    process_rus_kz(data_dir=data_dir,
                   out_dir=os.path.join(data_dir, out_dir),
                   img_dir="img",
                   gt_file=f"gt.txt")
