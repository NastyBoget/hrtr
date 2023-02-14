import json
import logging
import os
import shutil
import tempfile
import zipfile

import cv2
import pandas as pd
import wget

from process_datasets.abstract_dataset_processor import AbstractDatasetProcessor


class IMGUR5KDatasetProcessor(AbstractDatasetProcessor):

    __dataset_name = "imgur5k"
    __charset = '!"$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~¢£¥©«®°±²³µ·¹»¼½¾ÀÁÃÄÅÇÈÉËÍÏÑÖÜ' \
                'àáäåèéëïóö÷úûüýÿāăĆćČĒēėĜĝōőŞŪűŵŽȇʰʲʳʸ˚ˢˣ̇ΛΣΦΩβγδεηθλπρστυχϟᴬᴴᴹᴾᵀᵈᵏᵐᵗᵢᵣᶻẢếỌọ–—―‘’“”‟•…″⁎⁴⁵⁶⁷⁹⁺⁻⁽⁾ⁿ₀₁₂₃₉₊₌ₐₖₛ₤€℃℉™Ω⅛ⅠⅡⅢⅣⅤⅥ←' \
                '↑→↓↳↷↻⇒⇦⇾∂∅∆∇∈−∘√∝∞∫∮∴≈≠≡≤≥⊕⊖⊗⊘⋮①Ⓡ■□▲△▴▸►▼▾◂◆○●◦★☆☐♡⛤✓❖➁➂➜➝⟵⟶⟹⤷⤼⬑ⱼ〇〝〞・︶︿﹀﹄�𝒸'

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self.data_url = "https://at.ispras.ru/owncloud/index.php/s/ZnhtXmomnJgcK8X/download"
        self.logger = logger

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def charset(self) -> str:
        return self.__charset

    def process_dataset(self, out_dir: str, img_dir: str, gt_file: str) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            archive = os.path.join(data_dir, "archive.zip")
            self.logger.info(f"Downloading {self.dataset_name} dataset...")
            wget.download(self.data_url, archive)
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            data_dir = os.path.join(data_dir, "IMGUR5K")
            self.logger.info("Dataset downloaded")

            cropped_img_dir = os.path.join(data_dir, "cropped_img")
            os.makedirs(cropped_img_dir, exist_ok=True)
            df_dict = {}
            for stage in ["train", "test", "val"]:
                ann_path = os.path.join(data_dir, f"imgur5k_annotations_{stage}.json")
                df_dict[stage] = self.__process_json_annotations(ann_path, data_dir, cropped_img_dir, img_dir)

            char_set = set()
            for df in df_dict.values():
                for _, row in df.iterrows():
                    char_set = char_set | set(row["word"])
            self.logger.info(f"{self.dataset_name} char set: {repr(''.join(sorted(list(char_set))))}")
            self.__charset = char_set

            for stage, df in df_dict.items():
                df.to_csv(os.path.join(out_dir, f"{stage}_{gt_file}"), sep="\t", index=False, header=False)
            self.logger.info(f"{self.dataset_name} dataset length: train = {df_dict['train'].shape[0]}; val = {df_dict['val'].shape[0]}; test = {df_dict['test'].shape[0]}")  # noqa

            destination_img_dir = os.path.join(out_dir, img_dir)
            for img_name in os.listdir(cropped_img_dir):
                shutil.move(os.path.join(cropped_img_dir, img_name), os.path.join(destination_img_dir, img_name))

    def __process_json_annotations(self, annotations_path: str, base_dir: str, cropped_img_dir: str, img_dir: str) -> pd.DataFrame:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        result = {"path": [], "word": []}

        for img_id in annotations["index_id"]:
            img = cv2.imread(os.path.join(base_dir, annotations["index_id"][img_id]["image_path"]))
            if img is None:
                self.logger.info(f'Image {annotations["index_id"][img_id]["image_path"]} not found')
                continue

            for bbox_img_id in annotations["index_to_ann_map"][img_id]:
                try:
                    xc, yc, w, h, a = json.loads(annotations["ann_id"][bbox_img_id]['bounding_box'])
                    rotate_matrix = cv2.getRotationMatrix2D(center=(xc, yc), angle=-a, scale=1)
                    rotated_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img.shape[1], img.shape[0]))
                    crop_img = rotated_img[max(0, int(yc - h / 2)):int(yc + h / 2), max(0, int(xc - w / 2)):int(xc + w / 2)]
                    cv2.imwrite(os.path.join(cropped_img_dir, f"{bbox_img_id}.jpg"), crop_img)
                    result["path"].append(f"{img_dir}/{bbox_img_id}.jpg")
                    result["word"].append(annotations["ann_id"][bbox_img_id]['word'])
                except Exception as e:
                    self.logger.info(f"Error: {e}")

        return pd.DataFrame(result)
