# Handwritten Russian text recognition

## Datasets

| Name                                                                                                          | Size                                          |
|:--------------------------------------------------------------------------------------------------------------|:----------------------------------------------|
| [Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset) | train=72286, test=1544                        |
| [HKR](https://github.com/abdoelsayed2016/HKR_Dataset)                                                         | train=45470, val=9359, test1=5057, test2=5057 |

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
| Accuracy |  31.21   |   39.68   |   89.62   |  64.8   |
|   CER    |  22.17   |   39.32   |   2.92    |  20.9   |
|   WER    |  68.05   |   65.79   |   7.37    |  36.3   |


### With synthetic № 1

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  33.16   |   45.12   |   79.31   |  62.3   |
|   CER    |  19.36   |   18.92   |   4.13    |  11.4   |
|   WER    |  64.95   |   53.04   |   14.02   |  33.3   |


$$ HKR~all = \frac{test1 \cdot 4966 + test2 \cdot 5043}{10009} $$

## Results with preprocessing without augmentation

TODO

## Results with preprocessing and image augmentation

TODO

## Results with preprocessing and text augmentation

TODO

## Results with preprocessing and both text, image augmentation

TODO
