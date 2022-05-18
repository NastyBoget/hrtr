from typing import List

import cv2
import torch
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import detection_predictor

from model.dataset import AlignCollate
from model.model import Model
from model.utils import AttnLabelConverter
from utils import ModelOptions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HTRReader:

    def __init__(self, htr_model_path: str) -> None:
        self.model = detection_predictor(arch='db_resnet50', pretrained=True).eval()

        opt = ModelOptions(saved_model=htr_model_path, batch_size=1)
        self.htr_model = torch.nn.DataParallel(Model(opt)).to(device)
        self.htr_model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        self.htr_model.eval()

        self.align_converter = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.label_converter = AttnLabelConverter(opt.character)

    @staticmethod
    def _sort_bboxes(bboxes: List[tuple]) -> List[List[tuple]]:
        """
            x1, y1
            -------------------------
            |                       |
            |         Bbox          |
            |                       |
            -------------------------
                                x2, y2
        """
        lines_list = []
        bboxes = sorted(bboxes, key=lambda x: x[1])

        for bbox in bboxes:
            if len(lines_list) == 0:
                lines_list.append([bbox])
                continue
            iou_threshold = 0.4
            prev_bbox = lines_list[-1][-1]
            min_y1, max_y1 = min(bbox[1], prev_bbox[1]), max(bbox[1], prev_bbox[1])
            min_y2, max_y2 = min(bbox[3], prev_bbox[3]), max(bbox[3], prev_bbox[3])
            threshold = (min_y2 - max_y1) / (max_y2 - min_y1)
            if threshold >= iou_threshold:
                lines_list[-1].append(bbox)
            else:
                lines_list.append([bbox])
        lines_list = [sorted(line) for line in lines_list]

        return lines_list

    def _get_words_bboxes(self, doc_name: str) -> List[List[tuple]]:
        single_img_doc = DocumentFile.from_images(doc_name)
        out = self.model(single_img_doc)
        im = cv2.imread(doc_name)
        h, w, _ = im.shape
        bboxes = [(int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)) for box in out[0]]
        bboxes = self._sort_bboxes(bboxes)
        return bboxes

    def _recognize_word(self, img: Image.Image) -> str:
        batch_max_length = 25
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0).to(device)
        img_tensor, label_tensor = self.align_converter([(img, text_for_pred)])
        preds = self.htr_model(img_tensor, label_tensor, is_train=False)
        preds_size = torch.IntTensor([preds.size(1)] * 1)
        _, preds_index = preds.max(2)
        preds_str = self.label_converter.decode(preds_index.data, preds_size.data)
        preds_str = [pred[:pred.find('[s]')] for pred in preds_str]
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


if __name__ == "__main__":

    doc_path = "data/3.jpg"
    htr_reader = HTRReader(htr_model_path="saved_models/best_accuracy.pth")
    words_list = htr_reader.get_text(doc_path)
    for line in words_list:
        print(line)
