import logging
import os
from typing import Any, List, Optional

from torch.utils.data import ConcatDataset

from dataset.lmdb_dataset import LmdbDataset


def hierarchical_dataset(root: str, opt: Any, logger: logging.Logger, select_data: Optional[List[str]] = None) -> ConcatDataset:
    """
    Concatenate datasets of hierarchical dataset
    Select_data=None means getting all subdirectories of root directory
    """
    if select_data is None:
        select_data = ['/']

    dataset_list = []
    logger.info(f'Dataset root: {root}; dataset: {select_data[0]}')
    for dirpath, dirnames, filenames in os.walk(root):
        if dirnames:
            continue

        select_flag = False
        for selected_d in select_data:
            if selected_d in dirpath:
                select_flag = True
                break

        if select_flag:
            dataset = LmdbDataset(dirpath, opt, logger)
            logger.info(f'Subdirectory: {os.path.relpath(dirpath, root)}; num samples: {len(dataset)}')
            dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset
