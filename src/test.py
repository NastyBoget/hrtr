import os
import time

import torch
import torch.nn.functional as F
import torch.utils.data
from src.dataset.dataset import hierarchical_dataset, AlignCollate

from src.model.model import Model
from src.model.utils import CTCLabelConverter, AttnLabelConverter, Averager
from src.params import ModelOptions, russian_synthetic_char_set, russian_char_set, russian_kazakh_char_set
from src.utils.metrics import cer, wer, string_accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    confidence_score_list = []
    gt_list = []
    pred_list = []

    # Export predictions only when testing
    # because dir `/result/{opt.exp_name}` is created only during testing,
    # but this function is also be called by train.py.
    # Log file with predictions contains the dir of test data.
    if hasattr(opt, 'eval_data'):
        eval_dir = opt.eval_data.split("/")[-1]
        log_predictions = open(f'./result/{opt.exp_name}/log_predictions_{eval_dir}.txt', 'a')
        log_predictions.write(f'batch,target,prediction,match,cum_match\n')

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)
        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # Export predictions only when testing. This function is also be called by train.py.
            if hasattr(opt, 'eval_data'):
                log_predictions.write(f'{i},{gt},{pred},{int(pred == gt)},{n_correct}\n')

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            gt_list.append(gt)
            pred_list.append(pred)

    # Export predictions only when testing. This function is also be called by train.py.
    if hasattr(opt, 'eval_data'):
        log_predictions.close()

    return valid_loss_avg.val(), string_accuracy(pred_list, gt_list), cer(pred_list, gt_list), wer(pred_list, gt_list), \
        preds_str, confidence_score_list, labels, infer_time, length_of_data


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = "_".join(opt.saved_model.split('/')[-2:]).split(".")[-2]

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        _, accuracy, cer_value, wer_value, _, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt)
        log.write(eval_data_log)
        print(f'Accuracy: {accuracy:0.8f}')
        print(f'CER: {cer_value:0.8f}')
        print(f'WER: {wer_value:0.8f}')

        log.write(f'Accuracy: {accuracy:0.8f}\n')
        log.write(f'CER: {cer_value:0.8f}\n')
        log.write(f'WER: {wer_value:0.8f}\n')
        log.close()


if __name__ == '__main__':
    dataset = "hkr1"
    data_type = "hkr"
    metric = "accuracy"

    charset = None
    if "rus" in data_type:
        charset = russian_char_set
    else:
        charset = russian_kazakh_char_set
    if "synthetic" in data_type:
        charset = "".join(sorted(list(set(russian_synthetic_char_set + charset))))

    opt = ModelOptions(saved_model=f"/Users/anastasiabogatenkova/work/hrtr/saved_models/{data_type}/best_{metric}.pth",
                       character=charset)

    if dataset == "rus":
        opt.eval_data = "/Users/anastasiabogatenkova/work/hrtr/datasets/lmdb/test"
    elif dataset == "hkr1":
        opt.eval_data = "/Users/anastasiabogatenkova/work/hrtr/datasets/lmdb/test1"
    else:
        opt.eval_data = "/Users/anastasiabogatenkova/work/hrtr/datasets/lmdb/test2"

    test(opt)
