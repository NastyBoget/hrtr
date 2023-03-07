# Handwritten Russian text recognition

## TODO

- [ ] try [StackMix](https://github.com/ai-forever/StackMix-OCR)
- [ ] try [ScrabbleGAN](https://github.com/ai-forever/ScrabbleGAN)
- [ ] unify different models
- [ ] train on binarized data (optional)
- [ ] train on several datasets (optional)

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
