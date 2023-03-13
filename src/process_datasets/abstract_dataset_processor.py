from abc import ABC, abstractmethod
from typing import Optional


class AbstractDatasetProcessor(ABC):
    """
    This class is used for unify formats of different handwriting text datasets

    The output format should be:
    |
    |--- img_dir --- img_name_1
    |          |
    |          | --- ...
    |          |
    |          | --- img_name_n
    |
    |--- gt_files as tables with image name and text of the image
        |-------------------------------------|
        |         path        | text  | stage |
        |-------------------------------------|
        | img_dir/img_name_1 | text_1 | train |
        |--------------------|--------|-------|
        |                   ...               |
        |--------------------|--------|-------|
        | img_dir/img_name_n | text_n |  val  |
        |-------------------------------------|

    Gt file should have header and have "," as separator. Stages: train, val, test
    """

    @property
    @abstractmethod
    def charset(self) -> str:
        pass

    @abstractmethod
    def can_process(self, d_name: str) -> bool:
        pass

    @abstractmethod
    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str, dataset_name: str) -> None:
        """
        :param out_dir: directory path for saving images and groundtruth file
        :param img_dir: directory name inside out_dir for saving images
        :param gt_file: name of the groundtruth file
        :param dataset_name: name of the dataset
        """
        pass
