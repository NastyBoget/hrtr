import logging
from typing import List

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor
from process_datasets.cyrillic_processor import CyrillicDatasetProcessor
from process_datasets.hkr_processor import HKRDatasetProcessor
from process_datasets.synthetic_processor import SyntheticDatasetProcessor


def get_processors_list(logger: logging.Logger) -> List[AbstractDatasetProcessor]:
    processors = [
        SyntheticDatasetProcessor(logger=logger),
        CyrillicDatasetProcessor(logger=logger),
        HKRDatasetProcessor(logger=logger)
    ]
    return processors
