Handwritten Russian text recognition
====================================


First results
-------------

| Metric   | Cyrillic | HKR test1 | HKR test2 | HKR all |
| :------: | :------: | :-------: | :-------: | :-----: |
| Accuracy | 31.21    | 39.68     | 89.62     | 64.8    |
| CER      | 22.17    | 39.32     | 2.92      | 20.9    |
| WER      | 68.05    | 65.79     | 7.37      | 36.3    |


With synthetic
--------------

| Metric   | Cyrillic | HKR test1 | HKR test2 | HKR all |
| :------: | :------: | :-------: | :-------: | :-----: |
| Accuracy | 33.16    | 45.12     | 79.31     | 62.3    |
| CER      | 19.36    | 18.92     | 4.13      | 11.4    |
| WER      | 64.95    | 53.04     | 14.02     | 33.3    |


all = (test1 * 4966 + test2 * 5043) / (10009)


Attention-Based Fully Gated CNN-BGRU for Russian Handwritten Text (2020)
--------------

| Metric   | HKR test1 | HKR test2 | HKR all |
| :------: | :-------: | :-------: | :-----: |
| CER      | 4.13      | 6.31      | 4.5     |
| WER      | 18.91     | 23.69     | 19.2    |


StackMix and Blot Augmentations for Handwritten Text Recognition
---------------

| Metric   | HKR all |
| :------: | :-----: |
| Accuracy | 82.0    |
| CER      | 3.49    |
| WER      | 13.0    |
