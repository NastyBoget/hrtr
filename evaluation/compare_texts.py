import json
import os
import re
from tempfile import TemporaryDirectory
from typing import List, Dict

from texttable import Texttable


def _init_statistics_by_dataset(statistics: Dict, dataset_name: str) -> Dict:
    statistics[dataset_name] = {
        "Accuracy": [],
        "ASCII_Spacing_Characters": [],
        "ASCII_Special_Symbols": [],
        "ASCII_Digits": [],
        "ASCII_Uppercase_Letters": [],
        "Latin1_Special_Symbols": [],
        "Cyrillic": []
    }

    return statistics


def _update_statistics_by_symbol_kind(statistics_dataset: List, pattern: str, lines: List) -> List:
    matched = [line for line in lines if pattern in line]
    if matched:
        statistics_dataset.append(float(re.findall(r"\d+\.\d+", matched[0])[0][:-1]))

    return statistics_dataset


def _update_statistics_by_dataset(statistics: Dict, dataset: str, accuracy_path: str) -> Dict:
    statistic = statistics[dataset]
    with open(accuracy_path, "r") as f:
        lines = f.readlines()
        print(lines)
        matched = [line for line in lines if "Accuracy After Correction" in line]
        if not matched:
            matched = [line for line in lines if "Accuracy\n" in line]
        acc_percent = re.findall(r'\d+\.\d+', matched[0])[0][:-1]
        statistic["Accuracy"].append(float(acc_percent))

        statistic["ASCII_Spacing_Characters"] = _update_statistics_by_symbol_kind(statistic["ASCII_Spacing_Characters"],
                                                                                  "ASCII Spacing Characters",
                                                                                  lines)
        statistic["ASCII_Special_Symbols"] = _update_statistics_by_symbol_kind(statistic["ASCII_Special_Symbols"],
                                                                               "ASCII Special Symbols",
                                                                               lines)
        statistic["ASCII_Digits"] = _update_statistics_by_symbol_kind(statistic["ASCII_Digits"], "ASCII Digits", lines)
        statistic["ASCII_Spacing_Characters"] = _update_statistics_by_symbol_kind(statistic["ASCII_Spacing_Characters"],
                                                                                  "ASCII Spacing Characters",
                                                                                  lines)
        statistic["Cyrillic"] = _update_statistics_by_symbol_kind(statistic["Cyrillic"], "Cyrillic", lines)

    statistics[dataset] = statistic

    return statistics


def _get_avg(array: List) -> float:
    return sum(array) / len(array) if array else 0.0


def _get_avg_by_dataset(statistics: Dict, dataset: str) -> List:
    return [_get_avg(statistics[dataset]["ASCII_Spacing_Characters"]),
            _get_avg(statistics[dataset]["ASCII_Special_Symbols"]),
            _get_avg(statistics[dataset]["ASCII_Digits"]),
            _get_avg(statistics[dataset]["ASCII_Uppercase_Letters"]),
            _get_avg(statistics[dataset]["Latin1_Special_Symbols"]),
            _get_avg(statistics[dataset]["Cyrillic"]),
            _get_avg(statistics[dataset]["Accuracy"])]


if __name__ == "__main__":
    accs = [["Dataset", "Image name", "Accuracy OCR"]]
    accs_common = [["Dataset", "ASCII_Spacing_Chars", "ASCII_Special_Symbols", "ASCII_Digits",
                    "ASCII_Uppercase_Chars", "Latin1_Special_Symbols", "Cyrillic", "AVG Accuracy"]]

    statistics = {}
    data_dir = "../data/good_data"
    gt_file = "gt.json"
    pred_file = "pred.json"
    dataset_name = "good_data"

    statistics = _init_statistics_by_dataset(statistics, dataset_name)
    with open(os.path.join(data_dir, pred_file)) as f:
        pred = json.load(f)
    with open(os.path.join(data_dir, gt_file)) as f:
        gt = json.load(f)

    with TemporaryDirectory() as tmpdir:
        for img_name in gt:
            if img_name not in pred:
                print(f"{img_name} not found in gt")
                continue
            tmp_gt_path = os.path.join(tmpdir, f"{img_name}_tmp_gt.txt")
            tmp_htr_path = os.path.join(tmpdir, f"{img_name}_tmp_htr.txt")
            accuracy_path = os.path.join(tmpdir, f"{img_name}_accuracy.txt")
            with open(tmp_gt_path, "w") as f:
                print(gt[img_name], file=f)
            with open(tmp_htr_path, "w") as f:
                print(pred[img_name], file=f)
            try:
                # calculation accuracy build for Ubuntu from source https://github.com/eddieantonio/ocreval
                command = "./accuracy {} {} >> {}".format(tmp_gt_path, tmp_htr_path, accuracy_path)
                os.system(command)

                statistics = _update_statistics_by_dataset(statistics, dataset_name, accuracy_path)
                accs.append([dataset_name, img_name, statistics[dataset_name]["Accuracy"][-1]])
            except Exception as e:
                print(e)

    table_aacuracy_per_image = Texttable()
    table_aacuracy_per_image.add_rows(accs)

    # calculating average accuracy for each data set
    table_common = Texttable()

    for dataset_name in sorted(statistics.keys()):
        row = [dataset_name]
        row.extend(_get_avg_by_dataset(statistics, dataset_name))
        accs_common.append(row)
    table_common.add_rows(accs_common)

    with open("TPS-ResNet-BiLSTM-Attn-Seed1-Rus-Kz-Synth_result.txt", "w") as res_file:
        res_file.write("Table 1 - Accuracy for each file\n")
        res_file.write(table_aacuracy_per_image.draw())
        res_file.write("\n\nTable 2 - AVG by each type of symbols:\n")
        res_file.write(table_common.draw())

    print(table_aacuracy_per_image.draw())
    print(table_common.draw())
