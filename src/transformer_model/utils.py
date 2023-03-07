from typing import List, Union

import numpy as np
import random
import torch
import os
import editdistance
from warmup_scheduler import GradualWarmupScheduler


def set_seed(seed: int):
    """Set a random seed for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_model(config, model, epoch, train_loss, metric, optimizer, epochs_since_improvement, scheduler, scaler):
    '''Save PyTorch attention_model.'''

    torch.save({
        'attention_model': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'metric': metric,
        'optimizer': optimizer.state_dict(),
        'epochs_since_improvement': epochs_since_improvement,
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
    }, os.path.join(config.paths.save_dir, f'attention_model-{epoch}-{metric:.4f}.ckpt'))


def print_report(t, train_losses, metric, best_metric, lr):
    '''Print report of one epoch.'''
    print(f'Time: {t} s')
    for k, train_loss in train_losses.items():
        print(f'Train Loss [{k}]: {train_loss:.4f}')
    print(f'Current CER: {metric:.3f}')
    print(f'Best CER: {best_metric:.3f}')
    print(f'Learning Rate: {lr}')


def save_log(path, epoch, train_losses, metric):
    '''Save log of one epoch.'''
    with open(path, 'a') as file:
        file.write('epoch: ' + str(epoch))
        for k, train_loss in train_losses.items():
            file.write(f' train_loss [{k}]: {str(round(train_loss, 5))}')
        file.write(' metric: ' + str(round(metric, 5)))
        file.write('\n')


def character_error_rate(texts, text_preds):
    """CER count."""
    cers = []
    for p_seq1, p_seq2 in zip(texts, text_preds):
        if len(p_seq2) > 0:
            p_vocab = set(p_seq1 + p_seq2)
            p2c = dict(zip(p_vocab, range(len(p_vocab))))
            c_seq1 = [chr(p2c[p]) for p in p_seq1]
            c_seq2 = [chr(p2c[p]) for p in p_seq2]
            cers.append(editdistance.eval(''.join(c_seq1),
                                        ''.join(c_seq2)) / len(c_seq2))
        else:
            cers.append(1)
    return cers

def levenshtein_distance(first: Union[str, List[str]], second: Union[str, List[str]]) -> int:
    distance = [[0 for _ in range(len(second) + 1)] for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        for j in range(len(second) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                diag = distance[i - 1][j - 1] + (first[i - 1] != second[j - 1])
                upper = distance[i - 1][j] + 1
                left = distance[i][j - 1] + 1
                distance[i][j] = min(diag, upper, left)
    return distance[len(first)][len(second)]

def cer(pred_texts: List[str], gt_texts: List[str]) -> float:
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_chars = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        lev_distances += levenshtein_distance(pred_text, gt_text)
        num_gt_chars += len(gt_text)
    return lev_distances / num_gt_chars * 100.


def wer(pred_texts: List[str], gt_texts: List[str]) -> float:
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_words = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        gt_words, pred_words = gt_text.split(), pred_text.split()
        lev_distances += levenshtein_distance(pred_words, gt_words)
        num_gt_words += len(gt_words)
    return lev_distances / num_gt_words * 100.


def string_accuracy(pred_texts: List[str], gt_texts: List[str]) -> float:
    assert len(pred_texts) == len(gt_texts)
    correct = 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        correct += int(pred_text == gt_text)
    return correct / len(gt_texts) * 100.


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]