# M2-IPL
Modality-Missing RGBT Tracking Challenge

## Evaluation

Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1iIJYyz-TQqt0C97NITpLZDzgfVLrQRfk?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- URVIS
```shell
python tracking/test.py --dataset_name urvis
```
