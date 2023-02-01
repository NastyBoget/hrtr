from abc import ABC, abstractmethod


class DatasetProcessor(ABC):
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @abstractmethod
    def process_dataset(self, data_dir: str, out_dir: str, img_dir: str, gt_file: str) -> None:
        """
        :param data_dir: directory path with dataset
        :param out_dir: directory path for saving images and groundtruth file
        :param img_dir: directory name inside out_dir for saving images
        :param gt_file: name of the groundtruth file
        """
        pass
