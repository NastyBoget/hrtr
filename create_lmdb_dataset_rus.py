""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import shutil

import cv2
import lmdb
import numpy as np

from utils import check_valid_label

from process_datasets.merge_and_split_datasets import merge_datasets, split_datasets


def check_valid_image(image_bin: bytes):
    if image_bin is None:
        return False
    imageBuf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_dataset(input_path: str, gt_file: str, output_path: str) -> None:
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        input_path  : input folder path where starts imagePath
        output_path : LMDB output path
        gt_file     : list of image path and label
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gt_file, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        try:
            imagePath, label = datalist[i].strip('\n').split('\t')
            if not check_valid_label(label):
                print('%s is not a valid label' % label)
                continue
        except ValueError:
            continue
        imagePath = os.path.join(input_path, imagePath)

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        try:
            if not check_valid_image(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        except:
            print('error occured', i)
            with open(output_path + '/error_image_log.txt', 'a') as log:
                log.write('%s-th image data occured error\n' % str(i))
            continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    write_cache(env, cache)


if __name__ == "__main__":
    data_dir = "datasets"
    intermediate_dir = "merged"
    out_dir = "lmdb"
    image_dir = "img"
    os.makedirs(os.path.join(data_dir, intermediate_dir, image_dir), exist_ok=True)

    merge_datasets(data_dir=data_dir, img_dir=image_dir, out_dir=intermediate_dir)
    # TODO augmentation
    split_datasets(data_dir=data_dir, out_dir=intermediate_dir)

    for stage in ("train", "val", "test"):
        output_path = os.path.join(data_dir, out_dir, stage)
        os.makedirs(output_path, exist_ok=True)
        create_dataset(input_path=os.path.join(data_dir, intermediate_dir),
                       gt_file=os.path.join(data_dir, intermediate_dir, f"gt_{stage}.txt"),
                       output_path=output_path)
    shutil.rmtree(os.path.join(data_dir, intermediate_dir))
