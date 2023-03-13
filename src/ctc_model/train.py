import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import SequentialSampler

from src.ctc_model import utils
from src.ctc_model.ctc_labeling import CTCLabeling
from src.ctc_model.experiment import OCRExperiment
from src.ctc_model.model import get_ocr_model
from src.ctc_model.utils import kw_collate_fn
from src.dataset.ctc_dataset import CTCDataset
from src.dataset.transforms import transforms
from src.dataset.utils import get_charset
from src.utils.logger import get_logger

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--out_dir', help='Where to store models', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to the dataset', required=True)
    parser.add_argument('-n', '--label_files', nargs='+', required=True, help='Names of files with labels')
    parser.add_argument('--manual_seed', type=int, default=1111, help='For random seed setting')
    parser.add_argument('--batch_size', type=int, default=192, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--saved_model', type=str, default='', help="Path to attention_model to continue training")

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    os.makedirs(opt.out_dir, exist_ok=True)

    train_df_list, val_df_list = [], []
    opt.charset = ""

    for label_file in opt.label_files:
        data_df = pd.read_csv(os.path.join(opt.data_dir, label_file), sep=",", dtype={"text": str})
        opt.charset += get_charset(data_df)
        train_df_list.append(data_df[data_df.stage == "train"])
        val_df_list.append(data_df[data_df.stage == "val"])
    opt.charset = "".join(sorted(list(set(opt.charset))))

    logger.info('---------------------------------------- Options ----------------------------------------')
    args = vars(opt)
    for k, v in args.items():
        logger.info(f'{str(k)}: {str(v)}')
    logger.info('-----------------------------------------------------------------------------------------')

    utils.seed_everything(opt.manual_seed)
    ctc_labeling = CTCLabeling(opt.charset)

    train_dataset = CTCDataset(train_df_list, opt.data_dir, ctc_labeling, transforms)
    val_dataset = CTCDataset(val_df_list, opt.data_dir, ctc_labeling)

    model = get_ocr_model({'time_feature_count': 256, 'lstm_hidden': 256, 'lstm_len': 3, 'n_class': len(opt.charset) + 1}, pretrained=True)

    model = model.to(device)
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=kw_collate_fn, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=kw_collate_fn, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=opt.epochs, steps_per_epoch=len(train_loader), max_lr=0.001,
                                                    pct_start=0.1, anneal_strategy='cos', final_div_factor=10**5)

    logger.info('Start training')
    with open(os.path.join(opt.log_dir, opt.log_name), 'a+') as sys.stdout:
        if not opt.saved_model:
            experiment = OCRExperiment(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                base_dir=opt.out_dir,
                best_saving={'cer': 'min', 'wer': 'min', 'acc': 'max'},
                last_saving=True,
                low_memory=True,
                verbose_step=10**5,
                seed=opt.manual_seed,
                ctc_labeling=ctc_labeling,
            )
            experiment.fit(train_loader, val_loader, opt.epochs)
        else:
            logger.info('Resume training')
            experiment = OCRExperiment.resume(
                checkpoint_path=opt.saved_model,
                train_loader=train_loader,
                valid_loader=val_loader,
                n_epochs=opt.epochs,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                seed=opt.manual_seed,
                ctc_labeling=ctc_labeling,
            )
    experiment.destroy()
