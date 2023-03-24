import argparse
import gc
import time

import pandas as pd
import torch.nn.functional as F
from tabulate import tabulate
from torch.cuda import amp

from src.dataset.transformer_dataset import TransformerDataset
from src.dataset.transforms import transforms
from src.dataset.utils import get_charset
from src.transformer_model.custom_functions import MADGRAD
from src.transformer_model.data import CTCTokenizer, TransformerTokenizer, collate_fn
from src.transformer_model.model import CRNN
from src.transformer_model.utils import *
from src.utils.logger import get_logger
from src.utils.metrics import cer, wer, string_accuracy, levenshtein_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def val_loop(data_loader, model, tokenizers, logger):
    logger.info("Validation")
    final_predictions = {k: {'true': [], 'pred': []} for k in tokenizers.keys()}

    for data in data_loader:
        text_preds = predict(data['image'], data['image_mask'], model, tokenizers)
        for tokenizer_name, pred in text_preds.items():
            final_predictions[tokenizer_name]['true'].extend(data['text'])
            final_predictions[tokenizer_name]['pred'].extend(pred)

    cers = []
    wers = []
    accs = []
    for tokenizer_name, final_preds in final_predictions.items():
        df = pd.DataFrame(final_preds)
        df["cer"] = df.apply(lambda x: levenshtein_distance(x["pred"], x["true"]), axis=1)
        cer_value = cer(df["pred"], df["true"])
        wer_value = wer(df["pred"], df["true"])
        accuracy_value = string_accuracy(df["pred"], df["true"])

        print_df = df.sort_values(by='cer', ascending=False).iloc[:15]
        logger.info(f"Tokenizer {tokenizer_name}, CER = {cer_value}, WER = {wer_value}, accuracy = {accuracy_value}")
        logger.info(tabulate(print_df, headers='keys'))

        cers.append(cer_value)
        wers.append(wer_value)
        accs.append(accuracy_value)
    return min(cers), min(wers), max(accs)


def train_loop(data_loader, model, criterion_ctc, criterion_transformer, optimizer):
    losses = {}
    for name in 'ctc', 'transformer', 'total':
        losses[name] = AverageMeter()
    model.train()

    for data in data_loader:
        images = data['image'].to(device)
        image_masks = data['image_mask'].to(device)
        enc_text_transformer = data['enc_text_transformer'].to(device)

        model.zero_grad()
        output = model(images, image_masks, enc_text_transformer)

        output_lenghts = torch.full(size=(output['ctc'].size(1),), fill_value=output['ctc'].size(0), dtype=torch.long)
        alpha = 0.25
        loss_ctc = alpha * criterion_ctc(output['ctc'], data['enc_text_ctc'], output_lenghts, data['text_len'])

        transformer_expected = enc_text_transformer
        transformer_expected = F.pad(transformer_expected[:, 1:], pad=(0, 1, 0, 0), value=0)  # remove SOS token
        loss_transformer = (1 - alpha) * criterion_transformer(output['transformer'].permute(0, 2, 1), transformer_expected)

        loss = loss_transformer + loss_ctc
        losses['total'].update(loss.item(), len(data['text']))
        losses['ctc'].update(loss_ctc.item())
        losses['transformer'].update(loss_transformer.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    return {k: v.avg for k, v in losses.items()}


def predict(images, image_masks, model, tokenizers):
    model.eval()
    images = images.to(device)
    image_masks = image_masks.to(device)

    with torch.no_grad():
        output = model(images, image_masks, None)

    return {k: v.decode(output[k].detach().cpu()) for k, v in tokenizers.items()}


def run_train(opt, logger):
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

    tokenizer_ctc = CTCTokenizer(opt.charset)
    tokenizer_transformer = TransformerTokenizer(opt.charset)
    tokenizers = {'ctc': tokenizer_ctc, 'transformer': tokenizer_transformer}
    params = {'max_new_tokens': 30, 'min_length': 1, 'num_beams': 1, 'num_beam_groups': 1, 'do_sample': False}
    model = CRNN(n_ctc=tokenizer_ctc.get_num_chars(), n_transformer_decoder=tokenizer_transformer.get_num_chars(), transformer_decoding_params=params)
    model.to(device)

    scaler = amp.GradScaler()
    criterion_ctc = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    criterion_transformer = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = MADGRAD(model.parameters(), lr=1e-4 / 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.epochs - 5, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(optimizer=optimizer, multiplier=100, total_epoch=5, after_scheduler=scheduler)

    best_cer = np.inf
    early_stopping = 0
    start_epoch = 0

    if opt.saved_model:
        logger.info("Loading attention_model from checkpoint")
        cp = torch.load(opt.saved_model)

        scaler.load_state_dict(cp["scaler"])
        model.load_state_dict(cp["attention_model"])
        optimizer.load_state_dict(cp["optimizer"])
        for _ in range(cp["epoch"]):
            scheduler.step()

        early_stopping = cp["epochs_since_improvement"]
        start_epoch = cp["epoch"]
        del cp

    train_dataset = TransformerDataset(train_df_list, opt.data_dir, tokenizers, transforms)
    val_dataset = TransformerDataset(val_df_list, opt.data_dir, tokenizers)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, pin_memory=True)

    start_time = time.time()
    for epoch in range(start_epoch, opt.epochs):
        logger.info(f"Epoch: {epoch + 1}")

        train_loss = train_loop(train_loader, model, criterion_ctc, criterion_transformer, optimizer)
        cer_avg, wer_avg, acc_avg = val_loop(val_loader, model, tokenizers, logger)
        scheduler.step()

        t = int((time.time() - start_time) / 60.)
        if cer_avg < best_cer:
            logger.info("New record!")
            best_cer = cer_avg
            early_stopping = 0
            save_model(opt.out_dir, model, epoch + 1, train_loss, cer_avg, optimizer, early_stopping, scheduler, scaler)
        else:
            logger.info(f"Early stopping {early_stopping}/{opt.patience}")
            early_stopping += 1

        if early_stopping >= opt.patience:
            logger.info("Training has been interrupted because of early stopping")
            break

        for k, train_loss in train_loss.items():
            logger.info(f'Train Loss [{k}]: {train_loss:.4f}')
        logger.info(f'Current CER: {cer_avg:.3f}, current WER: {wer_avg:.3f}, current accuracy: {acc_avg:.3f}')
        logger.info(f'Best CER: {best_cer:.3f}')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]}, Elapsed time: {t} min')

        torch.cuda.empty_cache()
        gc.collect()


def run_eval(opt, logger):
    data_df = pd.read_csv(os.path.join(opt.data_dir, opt.label_files[0]), sep=",", dtype={"text": str})
    charset = get_charset(data_df)
    val_df = data_df[data_df.stage == opt.eval_stage]

    tokenizer_ctc = CTCTokenizer(charset)
    tokenizer_transformer = TransformerTokenizer(charset)
    tokenizers = {'ctc': tokenizer_ctc, 'transformer': tokenizer_transformer}

    params = {'max_new_tokens': 30, 'min_length': 1, 'num_beams': 1, 'num_beam_groups': 1, 'do_sample': False}
    model = CRNN(n_ctc=tokenizer_ctc.get_num_chars(), n_transformer_decoder=tokenizer_transformer.get_num_chars(), transformer_decoding_params=params)
    model.to(device)

    logger.info("Loading transformer model from checkpoint")

    cp = torch.load(opt.saved_model)
    model.load_state_dict(cp["attention_model"])
    del cp

    val_dataset = TransformerDataset([val_df], opt.data_dir, tokenizers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, pin_memory=True)

    cer_avg, wer_avg, acc_avg = val_loop(val_loader, model, tokenizers, logger)
    logger.info(f"Accuracy: {acc_avg}, CER: {cer_avg}, WER: {wer_avg}")
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='Directory for saving log file', required=True)
    parser.add_argument('--log_name', type=str, help='Name of the log file', required=True)
    parser.add_argument('--out_dir', help='Where to store models', default="transformer")
    parser.add_argument('--data_dir', type=str, help='Path to the dataset', required=True)
    parser.add_argument('-n', '--label_files', nargs='+', required=True, help='Names of files with labels')
    parser.add_argument('--manual_seed', type=int, default=1111, help='For random seed setting')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--saved_model', type=str, default='', help="Path to model to load")
    parser.add_argument('--patience', type=int, default=10, help='Patience for the early stopping')
    parser.add_argument('--eval_mode', action='store_true', help='Evaluation mode')
    parser.add_argument('--eval_stage', type=str, default='test', help='Name of test dataset stage')

    opt = parser.parse_args()
    set_seed(opt.manual_seed)

    os.makedirs(opt.log_dir, exist_ok=True)
    logger = get_logger(out_file=os.path.join(opt.log_dir, opt.log_name))
    os.makedirs(opt.out_dir, exist_ok=True)

    if opt.eval_mode:
        run_eval(opt, logger)
    else:
        start_time = time.time()
        run_train(opt, logger)
        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Overall elapsed time: {elapsed_time:.3f} min")
