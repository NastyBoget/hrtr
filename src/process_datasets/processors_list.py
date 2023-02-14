import logging
from typing import List

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor
from process_datasets.cyrillic_processor import CyrillicDatasetProcessor
from process_datasets.generate_processor import GenerateDatasetProcessor
from process_datasets.hkr_processor import HKRDatasetProcessor
from process_datasets.imgur5k_processor import IMGUR5KDatasetProcessor
from process_datasets.synthetic_processor import SyntheticDatasetProcessor


def get_processors_list(logger: logging.Logger) -> List[AbstractDatasetProcessor]:
    processors = [
        CyrillicDatasetProcessor(logger=logger),
        HKRDatasetProcessor(logger=logger),
        SyntheticDatasetProcessor(logger=logger),
        GenerateDatasetProcessor(logger=logger),
        IMGUR5KDatasetProcessor(logger=logger)
    ]
    return processors
