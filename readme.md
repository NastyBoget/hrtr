# Handwritten Russian text recognition

## Datasets

| Name                                                                                                          | Size                                          |
|:--------------------------------------------------------------------------------------------------------------|:----------------------------------------------|
| [Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset) | train=72286, test=1544                        |
| [HKR](https://github.com/abdoelsayed2016/HKR_Dataset)                                                         | train=45470, val=9359, test1=5057, test2=5057 |
| HKR (reality)                                                                                                 | train=45559, val=9375, test1=4966, test2=5043 |

HKR: val 9133 words in train, 242 words not in train

## Metrics

* Accuracy: $$accuracy=\frac{1}{N}\sum_{i=j}c_{ij}$$

* For characters: $$CER(prediction,real)=\frac{substitutions+insertions+deletes}{len(real)}$$

* For words: $$WER(prediction,real)=\frac{substitutions+insertions+deletes}{len(real)}$$

## Existing solutions

* [Attention-Based Fully Gated CNN-BGRU for Russian Handwritten Text (2020)](https://www.mdpi.com/2313-433X/6/12/141/htm)

| Metric | HKR test1 | HKR test2 | HKR all |
|:------:|:---------:|:---------:|:-------:|
|  CER   |   4.13    |   6.31    |   4.5   |
|  WER   |   18.91   |   23.69   |  19.2   |

* [StackMix and Blot Augmentations for Handwritten Text Recognition (2021)](https://arxiv.org/pdf/2108.11667)

|  Metric  | HKR all |
|:--------:|:-------:|
| Accuracy |  82.0   |
|   CER    |  3.49   |
|   WER    |  13.0   |

## Results without preprocessing and augmentation

### Without synthetic

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |   38.70   |   87.19   |  63.13  |
|   CER    |          |   40.63   |   3.45    |  21.89  |
|   WER    |          |   68.59   |   8.92    |  38.52  |


### With synthetic

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |           |           |         |
|   CER    |          |           |           |         |
|   WER    |          |           |           |         |


$$ HKR~all = \frac{test1 \cdot 4966 + test2 \cdot 5043}{10009} $$

## Results with preprocessing without augmentation

### Without synthetic

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |   38.02   |   86.30   |  62.34  |
|   CER    |          |   40.06   |   3.62    |  21.69  |
|   WER    |          |   69.45   |   9.58    |  39.28  |

### With synthetic

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |           |           |         |
|   CER    |          |           |           |         |
|   WER    |          |           |           |         |

## Results with preprocessing and image augmentation

### Without synthetic

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |   39.10   |   87.29   |  63.38  |
|   CER    |          |   38.63   |   3.22    |  20.78  |
|   WER    |          |   67.29   |   8.68    |  37.75  |

### With synthetic

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |           |           |         |
|   CER    |          |           |           |         |
|   WER    |          |           |           |         |

## Results with preprocessing and text augmentation

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |           |           |         |
|   CER    |          |           |           |         |
|   WER    |          |           |           |         |

## Results with preprocessing and both text, image augmentation

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |          |           |           |         |
|   CER    |          |           |           |         |
|   WER    |          |           |           |         |
