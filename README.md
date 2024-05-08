# PMFormer

This repository contains the supported pytorch code and configuration files to reproduce of PMFormer.

![PMFormer](img/Architecture_overview.png?raw=true)

Parts of codes are borrowed from [nn-UNet](https://github.com/MIC-DKFZ/nnUNet). For detailed configuration of the dataset, please refer to [nn-UNet](https://github.com/MIC-DKFZ/nnUNet).

## Environment

Please prepare an environment with Python 3.7, Pytorch 1.7.1, and Windows 10.

## Dataset Preparation

Datasets can be acquired via following links:

- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [The Synapse multi-organ CT dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- [Brain_tumor](http://medicaldecathlon.com/)
- [Heart](http://medicaldecathlon.com/)

## Dataset Set

After you have downloaded the datasets, you can follow the settings in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md) for path configurations and preprocessing procedures. Finally, your folders should be organized as follows:

```
./pmformer/
./DATASET/
  ├── PMFormer_raw/
      ├── PMFormer_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task04_Heart/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── PMFormer_cropped_data/
  ├── PMFormer_trained_models/
  ├── PMFormer_preprocessed/
```

## Preprocess Data

- PMFormer_convert_decathlon_task -i D:\Codes\Medical_image\UploadGitHub\PMFormer\DATASET\PMFormer_raw\PMFormer_raw_data
- PMFormer_plan_and_preprocess -t 2

## Functions of scripts

- **Network architecture:**
  - PMFormer\pmformer\network_architecture\PMFormer_acdc.py
  - PMFormer\pmformer\network_architecture\PMFormer_synapse.py
  - PMFormer\pmformer\network_architecture\PMFormer_tumor.py
  - PMFormer\pmformer\network_architecture\PMFormer_heart.py
- **Trainer for dataset:**
  - PMFormer\pmformer\training\network_training\PMFormerTrainerV2_pmformer_acdc.py
  - PMFormer\pmformer\training\network_training\PMFormerTrainerV2_pmformer_synapse.py
  - PMFormer\pmformer\training\network_training\PMFormerTrainerV2_pmformer_tumor.py
  - PMFormer\pmformer\training\network_training\PMFormerTrainerV2_pmformer_heart.py

## Train Model

- python run_training.py  3d_fullres  PMFormerTrainerV2_pmformer_synapse 2 0


## Test Model

- python predict.py -i D:\Codes\Medical_image\UploadGitHub\PMFormer\DATASET\PMFormer_raw\PMFormer_raw_data\Task002_Synapse\imagesTs
  -o D:\Codes\Medical_image\UploadGitHub\PMFormer\DATASET\PMFormer_raw\PMFormer_raw_data\Task002_Synapse\imagesTs_infer
  -m D:\Codes\Medical_image\UploadGitHub\PMFormer\DATASET\PMFormer_trained_models\PMFormer\3d_fullres\Task002_Synapse\PMFormerTrainerV2_PMFormer_synapse__PMFormerPlansv2.1
  -f 0

- python PMFormer/inference_synapse.py

## Acknowledgements

This repository makes liberal use of code from:

- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) 
- [nnFormer](https://github.com/282857341/nnFormer)
- [SSCFormer](https://github.com/YongChen-Exact/SSCFormer)
