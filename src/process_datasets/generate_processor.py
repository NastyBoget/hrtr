import logging

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor


class GenerateDatasetProcessor(AbstractDatasetProcessor):

    __dataset_name = "generate"
    __charset = ' !"%(),-.0123456789:;?АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя'

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def charset(self) -> str:
        return self.__charset

    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str) -> None:
        pass
