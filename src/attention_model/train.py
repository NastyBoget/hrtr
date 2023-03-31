""" modified version of deep-text-recognition-benchmark repository https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/train.py """
import argparse
import logging
import os
import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from src.attention_model.averager import Averager
from src.attention_model.label_converting import AttnLabelConverter, Converter
from src.attention_model.model import Model
from src.attention_model.resize_normalization import AlignCollate
from src.attention_model.test import validation
from src.dataset.attention_dataset import AttentionDataset
from src.dataset.transforms import transforms
from src.dataset.utils import get_charset
from src.utils.logger import get_logger
from src.utils.metrics import string_accuracy, cer, wer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data(opt: Any) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_df_list, val_df_list = [], []
    opt.character = ""

    for label_file in opt.label_files:
        data_df = pd.read_csv(os.path.join(opt.data_dir, label_file), sep=",", dtype={"text": str})
        opt.character += get_charset(data_df)
        train_df_list.append(data_df[data_df.stage == "train"])
        val_df_list.append(data_df[data_df.stage == "val"])
    opt.character = "".join(sorted(list(set(opt.character))))

    align_collate = AlignCollate(img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad)
    train_dataset = AttentionDataset(train_df_list, opt.data_dir, opt, transforms=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=align_collate, pin_memory=True)

    val_dataset = AttentionDataset(val_df_list, opt.data_dir, opt)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=align_collate, pin_memory=True)
    return train_loader, val_loader


def load_model(opt: Any, logger: logging.Logger) -> torch.nn.DataParallel:
    opt.num_class = len(opt.character) + 2
    model = Model(opt, logger)
    logger.info(f'Model input parameters: img_h={opt.img_h}, img_w={opt.img_w}, rgb={opt.rgb}')

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            logger.info(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)

    model = torch.nn.DataParallel(model)
    if opt.saved_model != '':
        logger.info(f'Loading pretrained attention_model from {opt.saved_model}')
        state_dict = torch.load(opt.saved_model, map_location=device)
        model.load_state_dict(state_dict, strict=not opt.ft)

    model.train()
    model = model.to(device)
    return model


def get_training_utils(logger: logging.Logger, model: torch.nn.DataParallel, opt: Any) -> Tuple[Converter, _Loss, Averager, Optimizer]:
    converter = AttnLabelConverter(opt.character)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_averager = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info(f'Trainable params num: {sum(params_num)}')
    optimizer = optim.Adadelta(filtered_parameters, lr=1, rho=0.95, eps=1e-8)

    logger.info('---------------------------------------- Options ----------------------------------------')
    args = vars(opt)
    for k, v in args.items():
        logger.info(f'{str(k)}: {str(v)}')
    logger.info('-----------------------------------------------------------------------------------------')
    return converter, criterion, loss_averager, optimizer


def train(opt: Any, logger: logging.Logger) -> None:
    train_loader, val_loader = prepare_data(opt)
    model = load_model(opt, logger)
    converter, criterion, loss_averager, optimizer = get_training_utils(logger, model, opt)

    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            logger.info(f'Continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time, best_accuracy, best_cer, iteration, best_loss, loss_increase_num = time.time(), -1, np.inf, start_iter, np.inf, 0
    epoch = 0

    while epoch < opt.epochs:

        for image_tensors, labels in train_loader:
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)

            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping with 5 (Default)
            optimizer.step()
            loss_averager.add(cost)
        epoch += 1

        # validation part
        elapsed_time = int((time.time() - start_time) / 60.)
        model.eval()
        with torch.no_grad():
            valid_loss, label_list, prediction_list, confidence_score_list = validation(model, criterion, val_loader, converter, opt, logger)
        model.train()
        current_accuracy, current_cer, current_wer = string_accuracy(prediction_list, label_list), cer(prediction_list, label_list), wer(prediction_list, label_list)  # noqa

        # training loss and validation loss
        train_loss = loss_averager.val()
        logger.info(f'[Epoch {epoch}/{opt.epochs}] Train loss: {train_loss:0.5f}, valid loss: {valid_loss:0.5f}, elapsed time: {elapsed_time} min')
        loss_averager.reset()
        logger.info(f'{"Current accuracy":17s}: {current_accuracy:0.3f}, {"current CER":17s}: {current_cer:0.2f}, {"current WER":17s}: {current_wer:0.2f}')
        best_accuracy = current_accuracy if current_accuracy > best_accuracy else best_accuracy
        # keep the best cer attention_model (on valid dataset)
        if current_cer < best_cer:
            best_cer = current_cer
            logger.info("Save attention_model with best CER")
            torch.save(model.state_dict(), os.path.join(opt.out_dir, 'best_cer.pth'))
        logger.info(f'{"Best accuracy":17s}: {best_accuracy:0.3f}, {"Best CER":17s}: {best_cer:0.2f}')

        # show some predicted results
        logger.info('-' * 84)
        logger.info(f'{"Ground Truth":^25s} | {"Prediction":^25s} | {"Confidence Score":^20s} | {"T/F":^5s}')
        logger.info('-' * 84)
        for gt, pred, confidence in zip(label_list[:5], prediction_list[:5], confidence_score_list[:5]):
            logger.info(f'{gt[:25]:^25s} | {pred[:25]:^25s} | {f"{confidence:0.4f}":^20s} | {str(pred == gt):^5s}')
        logger.info('-' * 84)

        if valid_loss < best_loss:
            logger.info(f"Validation loss decreased ({best_loss:0.5f} -> {valid_loss:0.5f}), saving attention_model...")
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(opt.out_dir, f'best_loss.pth'))
            loss_increase_num = 0
        else:
            loss_increase_num += 1
            logger.info(f"Validation loss increased {loss_increase_num}/{opt.patience}")
            if loss_increase_num > opt.patience:
                logger.info("Stop training (loss doesn't decrease)")
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--out_dir', help='Where to store models', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to the dataset', required=True)
    parser.add_argument('-n', '--label_files', nargs='+', required=True, help='Names of files with labels')
    parser.add_argument('--manual_seed', type=int, default=1111, help='For random seed setting')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--saved_model', type=str, default='', help="Path to attention_model to continue training")
    parser.add_argument('--ft', action='store_true', help='Whether to do fine-tuning')
    parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping')
    parser.add_argument('--write_errors', action='store_true', help='Write attention_model\'s errors to the log file')

    # Data processing
    parser.add_argument('--batch_max_length', type=int, default=40, help='Maximum label length')
    parser.add_argument('--img_h', type=int, default=32, help='The height of the input image')
    parser.add_argument('--img_w', type=int, default=100, help='The width of the input image')
    parser.add_argument('--rgb', action='store_true', help='Use rgb input')
    parser.add_argument('--pad', action='store_true', help='Whether to keep ratio then pad for image resize')

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    os.makedirs(opt.out_dir, exist_ok=True)

    # Seed and GPU setting
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    start_time = time.time()
    train(opt, logger)
    elapsed_time = (time.time() - start_time) / 60.
    logger.info(f"Overall elapsed time: {elapsed_time:.3f} min")
