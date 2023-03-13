import argparse
import os

import pandas as pd
import torch
from torch.utils.data import SequentialSampler

from src.ctc_model import utils
from src.ctc_model.ctc_labeling import CTCLabeling
from src.ctc_model.model import get_ocr_model
from src.ctc_model.predictor import Predictor
from src.ctc_model.utils import kw_collate_fn
from src.dataset.ctc_dataset import CTCDataset
from src.dataset.utils import get_charset
from src.utils.logger import get_logger
from src.utils.metrics import string_accuracy, cer, wer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--data_dir', type=str, help='Path to evaluation dataset', required=True)
    parser.add_argument('-n', '--label_files', nargs='+', required=True, help='Names of files with labels')
    parser.add_argument('--saved_model', type=str, help='Path to attention_model to evaluate', required=True)
    parser.add_argument('--write_errors', action='store_true', help='Write attention_model\'s errors to the log file')
    parser.add_argument('--batch_size', type=int, default=192, help='Input batch size')
    parser.add_argument('--manual_seed', type=int, default=1111, help='For random seed setting')
    parser.add_argument('--eval_stage', type=str, default='test', help='Name of test dataset stage')

    opt = parser.parse_args()

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    os.makedirs(opt.out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(opt.data_dir, opt.label_files[0]), sep=",", dtype={"text": str}, index_col='sample_id')
    charset = get_charset(df)
    opt.charset = charset

    utils.seed_everything(opt.manual_seed)

    ctc_labeling = CTCLabeling(charset)
    test_dataset = CTCDataset(df[df.stage == opt.eval_stage], opt.data_dir, ctc_labeling)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, collate_fn=kw_collate_fn, pin_memory=True)

    model = get_ocr_model({'time_feature_count': 256, 'lstm_hidden': 256, 'lstm_len': 3, 'n_class': len(charset) + 1}, pretrained=False)
    model = model.to(device)

    result_metrics = []

    logger.info(f'Loading pretrained model from {opt.saved_model}')
    checkpoint = torch.load(opt.saved_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predictor = Predictor(model, device)
    test_predictions = predictor.run_inference(test_loader)

    df_test_pred = pd.DataFrame([{
        'id': prediction['id'],
        'pred_text': ctc_labeling.decode(prediction['raw_output'].argmax(1)),
        'gt_text': prediction['gt_text']
    } for prediction in test_predictions]).set_index('id')

    if opt.write_errors:
        logger.info(f"{'-' * 40} Errors {'-' * 40}")
        logger.info(f'{"Ground Truth":^43s} | {"Prediction":^43s}')
        logger.info(f"{'-' * 88}")
        for _, row in df_test_pred.iterrows():
            logger.info(f'{row["gt_text"]:^43s} | {row["pred_text"]:^43s}')
        logger.info(f"{'-' * 88}")

    cer_value = cer(df_test_pred['pred_text'], df_test_pred['gt_text'])
    wer_value = wer(df_test_pred['pred_text'], df_test_pred['gt_text'])
    acc_value = string_accuracy(df_test_pred['pred_text'], df_test_pred['gt_text'])
    logger.info(f"Accuracy: {acc_value}, CER: {cer_value}, WER: {wer_value}")
