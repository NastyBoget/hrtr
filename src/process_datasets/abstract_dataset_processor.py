from abc import ABC, abstractmethod


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
        |---------------------|
        | img_name_1 | text_1 |
        |------------|--------|
        |           ...       |
        |------------|--------|
        | img_name_n | text_n |
        |---------------------|

    Names of gt_files should have prefix ending with _, e.g. train_gt_file, test_gt_file
    """

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @property
    @abstractmethod
    def charset(self) -> str:
        pass

    @abstractmethod
    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str) -> None:
        """
        :param out_dir: directory path for saving images and groundtruth file
        :param img_dir: directory name inside out_dir for saving images
        :param gt_file: name of the groundtruth file
        """
        pass
