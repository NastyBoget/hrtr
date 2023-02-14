""" modified version of deep-text-recognition-benchmark repository https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/train.py """
import argparse
import logging
import os
import random
import sys
import time
from typing import Any, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from src.dataset.batch_balanced_dataset import BatchBalancedDataset
from src.dataset.hierarchical_dataset import hierarchical_dataset
from src.dataset.preprocessing.resize_normalization import AlignCollate
from src.model.model import Model
from src.model.utils.averager import Averager
from src.model.utils.label_converting import CTCLabelConverter, AttnLabelConverter, Converter
from src.model.utils.metrics import string_accuracy, cer, wer
from src.process_datasets.processors_list import get_processors_list
from src.test import validation
from src.utils.logger import get_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_charset(opt: Any, logger: logging.Logger) -> str:
    processors = get_processors_list(logger)
    return "".join(sorted(list(set("".join(p.charset for p in processors if p.dataset_name in opt.select_data)))))


def prepare_data(opt: Any, logger: logging.Logger) -> Tuple[BatchBalancedDataset, torch.utils.data.DataLoader]:
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    if len(opt.initial_data) == 0:
        opt.character = get_charset(opt, logger)
    opt.character = opt.character if opt.sensitive else "".join(list(set(opt.character.lower())))

    train_dataset = BatchBalancedDataset(opt, logger)
    align_collate_valid = AlignCollate(img_h=opt.img_h, img_w=opt.img_w, keep_ratio_with_pad=opt.pad)
    augmentation = opt.augmentation
    opt.augmentation = False
    val_dataset = hierarchical_dataset(root=opt.valid_data, select_data=opt.select_data, opt=opt, logger=logger)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers), collate_fn=align_collate_valid, pin_memory=True)  # noqa
    opt.augmentation = augmentation
    return train_dataset, val_loader


def load_model(opt: Any, logger: logging.Logger) -> torch.nn.DataParallel:
    opt.num_class = len(opt.character) + 1 if 'CTC' in opt.prediction else len(opt.character) + 2
    opt.input_channel = 3 if opt.rgb else 1
    model = Model(opt)
    logger.info(f'Model input parameters: {opt.img_h}, {opt.img_w}, {opt.num_fiducial}, {opt.input_channel}, {opt.output_channel}, {opt.hidden_size},'
                f' {opt.num_class}, {opt.batch_max_length}, {opt.transformation}, {opt.feature_extraction}, {opt.sequence_modeling}, {opt.prediction}')

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
        logger.info(f'Loading pretrained model from {opt.saved_model}')
        if opt.ft:
            model.load_state_dict(torch.load(opt.saved_model, map_location=device), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))

    if len(opt.initial_data) > 0:
        char_set = get_charset(opt, logger)
        model.module.reset_output(charset=char_set)  # replace last layer to fine-tune model with another charset

    model.train()
    model = model.to(device)
    return model


def get_training_utils(logger: logging.Logger, model: torch.nn.DataParallel, opt: Any) -> Tuple[Converter, _Loss, Averager, Optimizer]:
    converter = CTCLabelConverter(opt.character) if 'CTC' in opt.prediction else AttnLabelConverter(opt.character)
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device) if 'CTC' in opt.prediction else torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_averager = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    logger.info(f'Trainable params num: {sum(params_num)}')
    optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999)) if opt.adam else optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)  # noqa

    logger.info('---------------------------------------- Options ----------------------------------------')
    args = vars(opt)
    for k, v in args.items():
        logger.info(f'{str(k)}: {str(v)}')
    logger.info('-----------------------------------------------------------------------------------------')
    return converter, criterion, loss_averager, optimizer


def train(opt: Any, logger: logging.Logger) -> None:
    model = load_model(opt, logger)
    train_dataset, val_loader = prepare_data(opt, logger)
    converter, criterion, loss_averager, optimizer = get_training_utils(logger, model, opt)

    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            logger.info(f'Continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time, best_accuracy, best_cer, iteration, best_loss, loss_increase_num = time.time(), -1, np.inf, start_iter, np.inf, 0

    n_train_samples = len(train_dataset.data_loader_list[0].dataset)
    iter_per_epoch = n_train_samples // opt.batch_size
    n_epochs = opt.num_iter // iter_per_epoch
    logger.info(f"Number of training samples: {n_train_samples}, number of epochs: {n_epochs}, iter per epoch: {iter_per_epoch}")
    opt.val_interval = iter_per_epoch
    epoch = 0

    while epoch < n_epochs:
    
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        loss_averager.add(cost)

        # validation part
        if (iteration + 1) % opt.val_interval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            model.eval()
            with torch.no_grad():
                valid_loss, label_list, prediction_list, confidence_score_list = validation(model, criterion, val_loader, converter, opt, logger)
            model.train()
            current_accuracy, current_cer, current_wer = string_accuracy(prediction_list, label_list), cer(prediction_list, label_list), wer(prediction_list, label_list)  # noqa

            # training loss and validation loss
            train_loss = loss_averager.val()
            logger.info(f'[{iteration + 1}/{opt.num_iter}] Train loss: {train_loss:0.5f}, valid loss: {valid_loss:0.5f}, elapsed time: {elapsed_time:0.5f}')
            loss_averager.reset()
            logger.info(f'{"Current accuracy":17s}: {current_accuracy:0.3f}, {"current CER":17s}: {current_cer:0.2f}, {"current WER":17s}: {current_wer:0.2f}')
            best_accuracy = current_accuracy if current_accuracy > best_accuracy else best_accuracy
            # keep best cer model (on valid dataset)
            if current_cer < best_cer:
                best_cer = current_cer
                logger.info("Save model with best CER")
                torch.save(model.state_dict(), os.path.join(opt.out_dir, 'best_cer.pth'))
            logger.info(f'{"Best accuracy":17s}: {best_accuracy:0.3f}, {"Best CER":17s}: {best_cer:0.2f}')

            # show some predicted results
            logger.info('-' * 84)
            logger.info(f'{"Ground Truth":^25s} | {"Prediction":^25s} | {"Confidence Score":^20s} | {"T/F":^5s}')
            logger.info('-' * 84)
            for gt, pred, confidence in zip(label_list[:5], prediction_list[:5], confidence_score_list[:5]):
                logger.info(f'{gt[:25]:^25s} | {pred[:25]:^25s} | {f"{confidence:0.4f}":^20s} | {str(pred == gt):^5s}')
            logger.info('-' * 84)
            
            # Check loss decrease at each epoch.
            if (iteration + 1) % iter_per_epoch == 0 or iteration == 0:
                logger.info(f"Iter: [{iteration + 1}/{opt.num_iter}]. Epoch: [{epoch}/{n_epochs}]. Training loss: {train_loss}.")
                if valid_loss < best_loss:
                    logger.info(f"Validation loss decreased ({best_loss:0.5f} -> {valid_loss:0.5f}), saving model...")
                    best_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(opt.out_dir, f'best_loss.pth'))
                    loss_increase_num = 0
                else:
                    loss_increase_num += 1
                    logger.info(f"Validation loss increased {loss_increase_num}/{opt.patience}")
                    if loss_increase_num > opt.patience:
                        logger.info("Stop training (loss doesn't decrease)")
                        break
                epoch += 1
              
        if (iteration + 1) % 5e+4 == 0:
            logger.info(f"Iteration {iteration + 1}, saving model...")
            torch.save(model.state_dict(), os.path.join(opt.out_dir, f'iter_{iteration + 1}.pth'))

        if (iteration + 1) == opt.num_iter:
            logger.info('End of the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--out_dir', help='Where to store models', required=True)
    parser.add_argument('--train_data', type=str, help='Path to training dataset', required=True)
    parser.add_argument('--valid_data', type=str, help='Path to validation dataset', required=True)
    parser.add_argument('--fonts_dir', type=str, help='Directory with fonts to generate images', default='fonts')
    parser.add_argument('--initial_data', type=str, help='Datasets of the pretrained model if we need to reset its output, separated by -', default='')
    parser.add_argument('--manual_seed', type=int, default=1111, help='For random seed setting')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=192, help='Input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='Number of iterations to train for')
    parser.add_argument('--val_interval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', type=str, default='', help="Path to model to continue training")
    parser.add_argument('--ft', action='store_true', help='Whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for adam, default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='Decay rate rho for Adadelta, default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='Eps for Adadelta, default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='Gradient clipping value, default=5')
    parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping')
    parser.add_argument('--write_errors', action='store_true', help='Write model\'s errors to the log file')

    # Data processing
    parser.add_argument('--select_data', type=str, help='Training data selection from datasets separated by -'
                                                        ' e.g. hkr-synthetic, use "generate" for text generation', required=True)
    parser.add_argument('--generate_num', type=int, help='The number of words to generate for each epoch', default=30000)
    parser.add_argument('--batch_ratio', type=str, help='Assign ratio for each selected data in the batch e.g. 0.5-0.5', required=True)
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0', help='Total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=40, help='Maximum label length')
    parser.add_argument('--img_h', type=int, default=32, help='The height of the input image')
    parser.add_argument('--img_w', type=int, default=100, help='The width of the input image')
    parser.add_argument('--rgb', action='store_true', help='Use rgb input')
    parser.add_argument('--sensitive', action='store_true', help='For sensitive character mode')
    parser.add_argument('--pad', action='store_true', help='Whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='For data_filtering_off mode')
    parser.add_argument('--add_generate_to_val', action='store_true', help='Add generated data to val dataset if text augmentation is used')
    parser.add_argument('--augmentation', action='store_true', help='Use augmentation during training')
    parser.add_argument('--preprocessing', action='store_true', help='Preprocess training data')

    # Model Architecture
    parser.add_argument('--transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--feature_extraction', type=str, required=True, help='Feature extraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--sequence_modeling', type=str, required=True, help='Sequence modeling stage. None|BiLSTM')
    parser.add_argument('--prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='Number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='The number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='The number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='The size of the LSTM hidden state')

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    os.makedirs(opt.out_dir, exist_ok=True)

    if len(opt.initial_data) > 0:
        opt.initial_data = opt.initial_data.split('-')
        processors = get_processors_list(logger)
        opt.character = "".join(sorted(list(set("".join(p.charset for p in processors if p.dataset_name in opt.initial_data)))))

    # Seed and GPU setting
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        logger.info('------ Use multi-GPU setting ------')
        logger.info('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

    train(opt, logger)
