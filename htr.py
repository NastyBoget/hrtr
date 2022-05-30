import json
import os
from typing import List

import cv2
import torch
from PIL import Image
from doctr.models import detection_predictor
from matplotlib import pyplot as plt
from tqdm import tqdm

from lines_segmentation.lines_seg_projections import sort_bboxes
from model.dataset import AlignCollate
from model.model import Model
from model.utils import AttnLabelConverter, CTCLabelConverter
from utils import ModelOptions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HTRReader:

    def __init__(self, opt: ModelOptions) -> None:
        self.model = detection_predictor(arch='db_resnet50', pretrained=True).eval()

        self.opt = opt
        self.htr_model = torch.nn.DataParallel(Model(self.opt)).to(device)
        self.htr_model.load_state_dict(torch.load(self.opt.saved_model, map_location=device))
        self.htr_model.eval()

        self.align_converter = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
        if self.opt.Prediction == "Attn":
            self.label_converter = AttnLabelConverter(self.opt.character)
        else:
            self.label_converter = CTCLabelConverter(self.opt.character)

    def _get_words_bboxes(self, doc_name: str) -> List[List[tuple]]:
        im = cv2.imread(doc_name)
        out = self.model([im])
        h, w, _ = im.shape
        bboxes = [(int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)) for box in out[0]]
        bboxes = sort_bboxes(im, bboxes)
        return bboxes

    def _recognize_word(self, img: Image.Image) -> str:
        batch_max_length = 25
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)
        img_tensor, label_tensor = self.align_converter([(img, text_for_pred)])
        if self.opt.Prediction == "Attn":
            preds = self.htr_model(img_tensor, label_tensor, is_train=False)
            preds_size = torch.IntTensor([preds.size(1)] * 1)
            _, preds_index = preds.max(2)
            preds_str = self.label_converter.decode(preds_index.data, preds_size.data)
            preds_str = [pred[:pred.find('[s]')] for pred in preds_str]
            return " ".join(preds_str)
        preds = self.htr_model(img_tensor, label_tensor)
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        preds_str = self.label_converter.decode(preds_index.data, preds_size.data)
        return " ".join(preds_str)

    def get_text(self, doc_name: str) -> List[List[str]]:
        bboxes = self._get_words_bboxes(doc_name)
        img = cv2.imread(doc_name, cv2.IMREAD_GRAYSCALE)
        lines_list = []
        for line in bboxes:
            words_list = []
            for box in line:
                x1, y1, x2, y2 = box
                cropped_img = img[y1:y2, x1:x2]
                cropped_img = Image.fromarray(cropped_img, 'L')
                words_list.append(self._recognize_word(cropped_img))
            lines_list.append(words_list)
        return lines_list


def save_results_img(in_dir: str, out_dir: str, reader: HTRReader) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for file_name in tqdm(os.listdir(in_dir)):
        if not file_name.endswith(".jpg"):
            continue
        words_list = reader.get_text(os.path.join(in_dir, file_name))
        words_list = [" ".join(line) for line in words_list]
        words_str = "\n".join(words_list)
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax.set_title(words_str)
        ax.axis('off')
        img = cv2.imread(os.path.join(in_dir, file_name))
        ax.imshow(img)
        fig.savefig(os.path.join(out_dir, file_name))
        plt.close(fig)


if __name__ == "__main__":
    save_results = False
    opt = ModelOptions(saved_model="saved_models/TPS-ResNet-BiLSTM-Attn-Seed1-Rus-Kz-Synth.pth",
                       batch_size=1, Prediction="Attn")
    htr_reader = HTRReader(opt)

    if save_results:
        save_results_img("data/lising", "data/result_lising", htr_reader)
    else:
        data_dir = "data/random_data"
        result = {}

        for file_name in tqdm(os.listdir(data_dir)):
            if not file_name.endswith(".jpg"):
                continue
            doc_path = os.path.join(data_dir, file_name)
            words_list = htr_reader.get_text(doc_path)
            words_list = [" ".join(line) for line in words_list]
            words_str = "\n".join(words_list)
            result[file_name] = words_str

        with open(os.path.join(data_dir, "pred.json"), "w") as f:
            json.dump(result, f)
