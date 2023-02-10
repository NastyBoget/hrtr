import logging
from typing import Any, Tuple, List

import torch
from torch._utils import _accumulate
from torch.utils.data import Subset

from src.dataset.hierarchical_dataset import hierarchical_dataset
from src.dataset.preprocessing.resize_normalization import AlignCollate
from src.dataset.text_generation_dataset import TextGenerationDataset


class BatchBalancedDataset(object):

    def __init__(self, opt: Any, logger: logging.Logger) -> None:
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ dataset and the other 50% of the batch is filled with ST dataset.
        """
        logger.info('=' * 100)
        logger.info(f'Dataset_root: {opt.train_data}; opt.select_data: {opt.select_data}; opt.batch_ratio: {opt.batch_ratio}')
        assert len(opt.select_data) == len(opt.batch_ratio)

        align_collate = AlignCollate(img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        total_batch_size = 0

        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            if selected_d == "generate":
                dataset = TextGenerationDataset(opt, logger=logger)
            else:
                dataset = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d], logger=logger)
            total_number_dataset = len(dataset)

            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            dataset, _ = [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            logger.info(f'Num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(dataset)}')
            batch_size_list.append(str(batch_size))
            total_batch_size += batch_size

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=align_collate,
                pin_memory=True
            )
            self.data_loader_list.append(data_loader)
            self.dataloader_iter_list.append(iter(data_loader))

        batch_size_sum = '+'.join(batch_size_list)
        opt.batch_size = total_batch_size
        logger.info(f'Total_batch_size: {batch_size_sum} = {total_batch_size}')
        logger.info('=' * 100)

    def get_batch(self) -> Tuple[torch.Tensor, List[str]]:
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts
