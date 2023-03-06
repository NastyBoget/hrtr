# Handwritten Russian text recognition

## TODO transformers

- [X] Use convnext_base
- [ ] Use runtime synthetic data generation
- [ ] Use preprocessing
- [ ] Use dataset with characters

## TODO AttentionHTR

- [ ] Use new binarization
- [ ] Use new augmentation
- [ ] Use dataset with characters

## TODO general

- [ ] try stackmix
- [ ] try adding kohtd dataset
- [ ] try training on all datasets
- [ ] try GANs? (ScrabbleGAN)
- [ ] try model from work about stackmix
- [ ] unify different models

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

## Results for AttentionHTR

### Results without preprocessing and augmentation

<details><summary>Without synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  34.45   |   38.70   |   87.19   |  63.13  |
|   CER    |  21.15   |   40.63   |   3.45    |  21.89  |
|   WER    |  64.95   |   68.59   |   8.92    |  38.52  |
</details>


<details><summary>With synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  29.33   |   41.72   |   78.02   |  60.00  |
|   CER    |  22.51   |   21.01   |   4.58    |  12.73  |
|   WER    |  68.22   |   55.18   |   15.12   |  34.99  |
</details>


<details><summary>With runtime generated data</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  37.50   |   43.19   |   83.81   |  63.65  |
|   CER    |  16.70   |   20.35   |   3.40    |  11.80  |
|   WER    |  58.46   |   52.50   |   11.09   |  31.63  |
</details>


<details><summary>With runtime generated data during training and validation</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  38.14   |   45.47   |   84.55   |  65.16  |
|   CER    |  16.35   |   18.20   |   3.22    |  10.65  |
|   WER    |  58.97   |   50.42   |   10.39   |  30.25  |
</details>


<details><summary>With pretrained model</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  32.18   |   43.36   |   76.61   |  60.11  |
|   CER    |  22.04   |   17.12   |   4.55    |  10.78  |
|   WER    |  67.40   |   51.23   |   16.18   |  33.57  |
</details>


$$ HKR~all = \frac{test1 \cdot 4966 + test2 \cdot 5043}{10009} $$


### Results with preprocessing without augmentation

<details><summary>Without synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  30.69   |   38.02   |   86.30   |  62.34  |
|   CER    |  24.84   |   40.06   |   3.62    |  21.69  |
|   WER    |  69.63   |   69.45   |   9.58    |  39.28  |
</details>


<details><summary>With synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  31.15   |   42.99   |   77.58   |  60.41  |
|   CER    |  21.56   |   19.68   |   4.79    |  12.17  |
|   WER    |  66.48   |   51.14   |   15.74   |  33.30  |
</details>


<details><summary>With runtime generated data</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  33.03   |   42.81   |   74.50   |  58.77  |
|   CER    |  20.66   |   17.39   |   5.47    |  11.38  |
|   WER    |  63.14   |   51.00   |   18.08   |  34.41  |
</details>


### Results with augmentation without preprocessing

<details><summary>Without synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  47.53   |   40.43   |   90.94   |  65.87  |
|   CER    |  14.87   |   42.67   |   2.32    |  22.33  |
|   WER    |  51.36   |   68.58   |   5.90    |  36.99  |
</details>


<details><summary>With synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  44.10   |   43.88   |   89.00   |  66.61  |
|   CER    |  14.28   |   25.44   |   2.46    |  13.86  |
|   WER    |  52.71   |   55.94   |   7.18    |  31.37  |
</details>


<details><summary>With runtime generated data</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  43.84   |   44.61   |   89.04   |  66.99  |
|   CER    |  14.45   |   23.67   |   2.49    |  12.99  |
|   WER    |  51.92   |   53.29   |   7.27    |  30.10  |
</details>


### Results with preprocessing and augmentation

<details><summary>Without synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  42.29   |   39.10   |   87.29   |  63.38  |
|   CER    |  17.80   |   38.63   |   3.22    |  20.78  |
|   WER    |  56.74   |   67.29   |   8.68    |  37.75  |
</details>


<details><summary>With synthetic</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  44.55   |   43.66   |   86.06   |  65.02  |
|   CER    |  15.62   |   22.01   |   3.12    |  12.49  |
|   WER    |  51.83   |   53.27   |   9.27    |  31.10  |
</details>


<details><summary>With runtime generated data</summary>

|  Metric  | Cyrillic | HKR test1 | HKR test2 | HKR all |
|:--------:|:--------:|:---------:|:---------:|:-------:|
| Accuracy |  44.43   |   47.63   |   86.06   |  66.99  |
|   CER    |  15.63   |   19.03   |   3.12    |  11.01  |
|   WER    |  52.94   |   48.30   |   9.27    |  28.63  |
</details>
