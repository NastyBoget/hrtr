"""
Modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py

Input:                                  Output:
    |                                         |
    |--- img_dir --- img_name_1               |--- train --- dataset_1 --- lmdb_files
    |          |                              |        |
    |          | --- ...                      |        | --- dataset_n --- lmdb_files
    |          |                              |
    |          | --- img_name_n               |--- test (same as train)
    |                                         |
    |--- gt_file                              |--- val (same as train)

"""
import argparse
import logging
import os
import tempfile
from typing import Any

import cv2
import lmdb
import numpy as np
import pandas as pd
from lmdb import Environment

from process_datasets.cyrillic_processor import CyrillicDatasetProcessor
from process_datasets.hkr_processor import HKRDatasetProcessor
from process_datasets.synthetic_processor import SyntheticDatasetProcessor
from utils.logger import get_logger


def check_valid_label(label: str, char_set: str) -> bool:
    return set(label).issubset(char_set)


def check_valid_image(image_bin: bytes) -> bool:
    if image_bin is None:
        return False
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h * img_w == 0:
        return False
    return True


def write_cache(env: Environment, cache: dict) -> None:
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_dataset(input_path: str, gt_file: str, output_path: str, char_set: str, logger: logging.Logger) -> None:
    """
    Create LMDB dataset for training and evaluation.
    :param input_path: input folder path where starts path to the images
    :param gt_file: path of the file with list of images' names and labels
    :param output_path: LMDB output path
    :param char_set: set of allowed characters
    :param logger: logger
    """
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1

    df = pd.read_csv(gt_file, sep="\t", names=["path", "word"])

    n_samples = df.shape[0]
    for index, row in df.iterrows():
        try:
            image_path, label = row["path"], row["word"]
            if not check_valid_label(label, char_set):
                logger.info(f'{label} is not a valid label')
                continue

            image_path = os.path.join(input_path, image_path)
            if not os.path.exists(image_path):
                logger.info(f'{image_path} does not exist')
                continue

            with open(image_path, 'rb') as f:
                image_bin = f.read()

            if not check_valid_image(image_bin):
                logger.info(f'{image_path} is not a valid image')
                continue
        except Exception as e:
            logger.error(f'Error occurred on {index} iteration: {e}')
            continue

        cache[f'image-{cnt:09d}'.encode()] = image_bin
        cache[f'label-{cnt:09d}'.encode()] = label.encode()

        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            logger.info(f'Written {cnt} / {n_samples}')
        cnt += 1
    n_samples = cnt - 1
    logger.info(f"Resulting number of samples: {n_samples}")
    cache['num-samples'.encode()] = str(n_samples).encode()
    write_cache(env, cache)


def create_lmdb_dataset(options: Any) -> None:
    os.makedirs(options.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(options.log_dir, options.log_name))

    dataset_processors = [
        SyntheticDatasetProcessor(logger=logger),
        CyrillicDatasetProcessor(logger=logger),
        HKRDatasetProcessor(logger=logger)
    ]

    image_dir = "img"
    os.makedirs(options.out_dir, exist_ok=True)

    for dataset_processor in dataset_processors:
        if dataset_processor.dataset_name not in options.datasets_list:
            continue

        logger.info("=" * 100)
        logger.info(f"Processing {dataset_processor.dataset_name} dataset")
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, image_dir))
            dataset_processor.process_dataset(tmpdir, image_dir, f"{dataset_processor.dataset_name}_gt.txt")

            for gt_file_name in os.listdir(tmpdir):
                if not gt_file_name.endswith("gt.txt"):
                    continue

                gt_prefix = gt_file_name.split("_")[0]
                subdataset_dir = os.path.join(options.out_dir, gt_prefix, dataset_processor.dataset_name)
                os.makedirs(subdataset_dir)
                logger.info(f"Create LMDB {gt_prefix}")
                create_dataset(tmpdir, os.path.join(tmpdir, gt_file_name), subdataset_dir, dataset_processor.charset, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--datasets_list', nargs='+', required=True, help='List of datasets names, e.g. hkr cyrillic')
    parser.add_argument('--out_dir', type=str, help='Directory for saving dataset', required=True)
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)

    opt = parser.parse_args()
    create_lmdb_dataset(opt)
