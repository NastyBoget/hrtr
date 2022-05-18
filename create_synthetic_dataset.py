import os
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from doctr.models import detection_predictor
from tqdm import tqdm

model = detection_predictor(arch='db_resnet50', pretrained=True).eval()


def get_empty_line_y(pr: np.ndarray, threshold: int) -> List[int]:
    """
    |-------------------
    |first written line
    |
    |-------------------    empty line (we need to find its coordinate for each pair of adjacent lines)
    |
    |second written line
    |-------------------

    :param threshold: the number of pixels which can be empty in one line
    :param pr: image projection on the y axis
    :return: list of y - middle line coordinate between two lines
    """
    y_list = []
    y_start = 0

    for i, s in enumerate(pr):
        if s != 0:
            continue
        if s == 0 and i > 0 and pr[i - 1] != 0:
            y_start = i
            continue
        if s == 0 and i < len(pr) - 1 and pr[i + 1] != 0 and i - y_start > threshold:
            y_list.append((i + y_start) // 2)

    y_list.append(len(pr) - 1)
    return y_list


def save_words_bboxes(dir_path: str, out_path: str) -> Optional[pd.DataFrame]:
    """
    1. Read the list of words - each word is on the separate line

    2. Sort document pages

    3. Find the list of coordinates of separating empty lines for each page
    lists = list1, list2, ..., listn
    the length of each list is equal the number of words in a page + 1

    4. Check if the number of words in all pages is equal the number of words in txt file
    sum([length(l) - 1 for each l in lists]) = length(text)

    5. Detect text bounding boxes for each page, check the boundaries of y1, y2 for each bbox usind lists

    6. Save cropped images if everything is correct and connect pictures' names with the words in text

    :param dir_path: path to the dir with pages images and handwrittner.com.txt file with the text
    :param out_path: path for saving cropped images
    :return: dataset [word path] for the given list of pages
    """
    with open(os.path.join(dir_path, "handwrittner.com.txt"), "r") as f:
        text = f.read()
    text = text.strip().split()  # list of words
    dir_name = os.path.split(dir_path)[-1]

    img_list = os.listdir(dir_path)
    img_list = [img for img in img_list if img.endswith(".png")]
    img_list = [(int(img.split("-")[0]), img) for img in img_list]
    img_list = sorted(img_list)
    img_list = [item[1] for item in img_list]

    y_lists = []
    for threshold in range(15):
        y_lists = []
        for img_name in img_list:
            img = cv2.imread(os.path.join(dir_path, img_name))
            pr = img.sum(axis=1).sum(axis=1)
            pr = np.max(pr) - pr
            y_lists.append(get_empty_line_y(pr, threshold))

        if np.sum([len(y_list) - 1 for y_list in y_lists]) == len(text):
            break

        if not (np.sum([len(y_list) - 1 for y_list in y_lists]) == len(text)) and threshold == 14:
            print(f"{dir_name} the number of words images is not equal to the number words")
            return

    images_names = [[] for _ in y_lists]
    for img_num, img_name in enumerate(img_list):
        current_y_list = y_lists[img_num]
        detected_bounds = [[] for _ in range(len(current_y_list) - 1)]

        img = cv2.imread(os.path.join(dir_path, img_name))
        out = model([img])
        h, w, _ = img.shape
        bboxes = [(int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)) for box in out[0]]

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            for y_i in range(len(current_y_list) - 1):
                if current_y_list[y_i] <= y1 <= y2 <= current_y_list[y_i + 1]:
                    detected_bounds[y_i].append(bbox)

        for i, item in enumerate(detected_bounds):
            if len(item) != 1:
                detected_bounds[i] = []

        for i, bbox_list in enumerate(detected_bounds):
            if len(bbox_list) == 0:
                images_names[img_num].append("")
                continue
            x1, y1, x2, y2 = bbox_list[0]
            cropped_img = img[y1:y2, x1:x2]
            cropped_img = Image.fromarray(cropped_img)
            cropped_img_name = f"{dir_name}_{img_num}_{i}.png"
            images_names[img_num].append(f"img/{cropped_img_name}")
            cropped_img.save(os.path.join(out_path, cropped_img_name))

    df = pd.DataFrame({"path": [img_name for img_page in images_names for img_name in img_page], "word": text})
    df = df[df.path != ""]
    return df


if __name__ == "__main__":
    dataset_dir = os.path.join("datasets", "synthetic")
    raw_dir = os.path.join("datasets", "synthetic", "raw")
    out_dir = "cropped"
    os.makedirs(os.path.join(dataset_dir, out_dir), exist_ok=True)
    df_list = []

    for dir_name in tqdm(os.listdir(raw_dir)):
        if not dir_name.startswith("handwrittner"):
            continue
        dir_path = os.path.join(raw_dir, dir_name)
        new_df = save_words_bboxes(dir_path, os.path.join(dataset_dir, out_dir))
        if new_df is not None:
            df_list.append(new_df)
        print(f"{dir_name} processed")

    if len(df_list) > 0:
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(os.path.join(dataset_dir, "gt.txt"), sep="\t", index=False, header=False)
    # 27 33 34 37 14 11 20 TODO
