import argparse
import logging
import os
from typing import Any, Tuple, List

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn.modules.loss import _Loss

from src.attention_model.averager import Averager
from src.attention_model.label_converting import Converter, AttnLabelConverter
from src.attention_model.model import Model
from src.attention_model.resize_normalization import AlignCollate
from src.dataset.attention_dataset import AttentionDataset
from src.dataset.utils import get_charset
from src.utils.logger import get_logger
from src.utils.metrics import string_accuracy, cer, wer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model: torch.nn.DataParallel,
               criterion: _Loss,
               evaluation_loader: torch.utils.data.DataLoader,
               converter: Converter,
               opt: Any,
               logger: logging.Logger) -> Tuple[float, List[str], List[str], List[float]]:
    valid_loss_avg = Averager()
    confidence_score_list, label_list, prediction_list = [], [], []

    if opt.write_errors:
        logger.info(f"{'-' * 40} Errors {'-' * 40}")
        logger.info(f'{"Ground Truth":^43s} | {"Prediction":^43s}')
        logger.info(f"{'-' * 88}")

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        preds = model(image, text_for_pred, is_train=False)
        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:]  # without [GO] Symbol
        cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        valid_loss_avg.add(cost)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            gt = gt[:gt.find('[s]')]
            pred_eos = pred.find('[s]')
            pred = pred[:pred_eos]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_eos]

            if opt.write_errors and gt != pred:
                logger.info(f'{gt:^43s} | {pred:^43s}')

            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            label_list.append(gt)
            prediction_list.append(pred)

    if opt.write_errors:
        logger.info(f"{'-' * 88}")

    return valid_loss_avg.val(), label_list, prediction_list, confidence_score_list


def test(opt: Any, logger: logging.Logger) -> None:
    label_file = opt.label_files[0]
    data_df = pd.read_csv(os.path.join(opt.data_dir, label_file), sep=",")
    opt.character = get_charset(data_df)
    test_df = data_df[data_df.stage == opt.eval_stage]
    converter = AttnLabelConverter(opt.character)
    align_collate = AlignCollate(img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad)
    opt.num_class = len(converter.character)
    test_dataset = AttentionDataset([test_df], opt.data_dir, opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, collate_fn=align_collate, pin_memory=True)

    model = Model(opt, logger)
    model = torch.nn.DataParallel(model).to(device)
    logger.info(f'Loading pretrained model from {opt.saved_model}')
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device) # noqa
    model.eval()

    with torch.no_grad():
        _, label_list, prediction_list, _ = validation(model, criterion, test_loader, converter, opt, logger)
        logger.info(f'Accuracy: {string_accuracy(prediction_list, label_list):0.8f}, CER: {cer(prediction_list, label_list):0.8f}, WER: {wer(prediction_list, label_list):0.8f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to dataset', required=True)
    parser.add_argument('-n', '--label_files', nargs='+', required=True, help='Names of files with labels')
    parser.add_argument('--saved_model', type=str, help='Path to attention_model to evaluate', required=True)
    parser.add_argument('--write_errors', action='store_true', help='Write attention_model\'s errors to the log file')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--eval_stage', type=str, default='test', help='Name of test dataset stage')

    # Data processing
    parser.add_argument('--batch_max_length', type=int, default=40, help='Maximum label length')
    parser.add_argument('--img_h', type=int, default=32, help='The height of the input image')
    parser.add_argument('--img_w', type=int, default=100, help='The width of the input image')
    parser.add_argument('--rgb', action='store_true', help='Use rgb input')
    parser.add_argument('--pad', action='store_true', help='Whether to keep ratio then pad for image resize')

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    test(opt, logger)
