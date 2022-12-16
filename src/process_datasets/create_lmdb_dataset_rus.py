""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import argparse
import os
import shutil

import cv2
import lmdb
import numpy as np

from src.params import check_valid_label, charsets
from src.process_datasets.merge_and_split_datasets import merge_datasets


def check_valid_image(image_bin: bytes):
    if image_bin is None:
        return False
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h * img_w == 0:
        return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_dataset(input_path: str, gt_file: str, output_path: str, char_set: str) -> None:
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        input_path  : input folder path where starts imagePath
        gt_file     : list of image path and label
        output_path : LMDB output path
        char_set : set of allowed characters
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gt_file, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    n_samples = len(datalist)
    for i in range(n_samples):
        try:
            image_path, label = datalist[i].strip('\n').split('\t')
            if not check_valid_label(label, char_set):
                print(f'{label} is not a valid label')
                continue
        except ValueError:
            continue
        image_path = os.path.join(input_path, image_path)

        if not os.path.exists(image_path):
            print(f'{image_path} does not exist')
            continue
        with open(image_path, 'rb') as f:
            image_bin = f.read()
        try:
            if not check_valid_image(image_bin):
                print(f'{image_path} is not a valid image')
                continue
        except Exception as e:
            print(f'error occurred on {i} iteration: {e}')
            with open(output_path + '/error_image_log.txt', 'a') as log:
                log.write(f'{i}-th image data occurred error: {e}\n')
            continue

        image_key = 'image-%09d'.encode() % cnt
        label_key = 'label-%09d'.encode() % cnt
        cache[image_key] = image_bin
        cache[label_key] = label.encode()

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print(f'Written {cnt} / {n_samples}')
        cnt += 1
    n_samples = cnt-1
    cache['num-samples'.encode()] = str(n_samples).encode()
    write_cache(env, cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--datasets_list', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, help='directory for saving dataset', required=True)

    opt = parser.parse_args()

    data_dir = opt.out_dir
    intermediate_dir = "merged"
    out_dir = "lmdb"
    image_dir = "img"
    os.makedirs(os.path.join(data_dir, intermediate_dir, image_dir), exist_ok=True)

    datasets_list = opt.datasets_list
    merge_datasets(data_dir=data_dir, img_dir=image_dir, out_dir=intermediate_dir, datasets_list=datasets_list)
    char_set = "".join(set("".join(charsets[dataset_name] for dataset_name in datasets_list)))

    for stage in ("train", "val", "test"):
        output_path = os.path.join(data_dir, out_dir, stage)
        os.makedirs(output_path, exist_ok=True)
        create_dataset(input_path=os.path.join(data_dir, intermediate_dir),
                       gt_file=os.path.join(data_dir, intermediate_dir, f"gt_{stage}.txt"),
                       output_path=output_path,
                       char_set=char_set)
    shutil.rmtree(os.path.join(data_dir, intermediate_dir))
