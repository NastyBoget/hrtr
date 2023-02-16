import argparse
import logging
import os
from typing import Any, Tuple, List

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.nn.modules.loss import _Loss

from src.dataset.hierarchical_dataset import hierarchical_dataset
from src.dataset.preprocessing.resize_normalization import AlignCollate
from src.model.model import Model
from src.model.utils.averager import Averager
from src.model.utils.label_converting import CTCLabelConverter, AttnLabelConverter, Converter
from src.model.utils.metrics import cer, wer, string_accuracy
from src.process_datasets.processors_list import get_processors_list
from src.utils.logger import get_logger

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

        if 'CTC' in opt.prediction:
            preds = model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
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
            if 'Attn' in opt.prediction:
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
    opt.select_data = opt.select_data.split('-')
    opt.character = "".join(sorted(list(set("".join(p.charset for p in get_processors_list(logger) if p.dataset_name in opt.select_data)))))
    converter = CTCLabelConverter(opt.character) if 'CTC' in opt.prediction else AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    opt.input_channel = 3 if opt.rgb else 1

    model = Model(opt, logger)
    logger.info(f'Model input parameters: {opt.img_h}, {opt.img_w}, {opt.num_fiducial}, {opt.input_channel}, {opt.output_channel}, {opt.hidden_size},'
                f' {opt.num_class}, {opt.batch_max_length}, {opt.transformation}, {opt.feature_extraction}, {opt.sequence_modeling}, {opt.prediction}')
    model = torch.nn.DataParallel(model).to(device)
    logger.info(f'Loading pretrained model from {opt.saved_model}')
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device) if 'CTC' in opt.prediction else torch.nn.CrossEntropyLoss(ignore_index=0).to(device) # noqa
    model.eval()

    with torch.no_grad():
        align_collate_evaluation = AlignCollate(img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad)
        eval_data = hierarchical_dataset(root=opt.test_data, opt=opt, logger=logger)
        evaluation_loader = torch.utils.data.DataLoader(eval_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, collate_fn=align_collate_evaluation, pin_memory=True)  # noqa
        _, label_list, prediction_list, _ = validation(model, criterion, evaluation_loader, converter, opt, logger)
        logger.info(f'Accuracy: {string_accuracy(prediction_list, label_list):0.8f}, CER: {cer(prediction_list, label_list):0.8f}, WER: {wer(prediction_list, label_list):0.8f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--test_data', type=str, help='Path to evaluation dataset', required=True)
    parser.add_argument('--saved_model', type=str, help='Path to model to evaluate', required=True)
    parser.add_argument('--select_data', type=str, help='List of datasets on which model was trained separated by -, e.g. hkr-synthetic', required=True)  # noqa
    parser.add_argument('--write_errors', action='store_true', help='Write model\'s errors to the log file')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=192, help='Input batch size')

    # Data processing
    parser.add_argument('--batch_max_length', type=int, default=40, help='Maximum label length')
    parser.add_argument('--img_h', type=int, default=32, help='The height of the input image')
    parser.add_argument('--img_w', type=int, default=100, help='The width of the input image')
    parser.add_argument('--rgb', action='store_true', help='Use rgb input')
    parser.add_argument('--sensitive', action='store_true', help='For sensitive character mode')
    parser.add_argument('--pad', action='store_true', help='Whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='For data_filtering_off mode')
    parser.add_argument('--preprocessing', action='store_true', help='Preprocess training data')

    # Model Architecture
    parser.add_argument('--transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--feature_extraction', type=str, required=True, help='Feature extraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--sequence_modeling', type=str, required=True, help='Sequence modeling stage. None|BiLSTM|BiGRU')
    parser.add_argument('--prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='Number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='The number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='The number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='The size of the LSTM hidden state')

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    test(opt, logger)
