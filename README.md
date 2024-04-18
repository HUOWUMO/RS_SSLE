# Non-parametric Spectral-Spatial Augmentation with Invariant Learning for Cross-Scene Hyperspectral Image Classificationn

<p align='center'>
  <img src='abstract_00.png' width="800px">
</p>

## Abstract

For hyperspectral cross-scene classification tasks involving invisible target domains (TDs), the model is trained only in the source domains (SDs) and then applied directly to the TDs. The main challenges in this domain generalization problem are the variations in reflection spectrum due to various imaging conditions. To address the discrepancy between SD and TD, this paper proposes a non-parametric strategy for domain expansion using spectral-spatial augmentation. Weighted resampling in the spatial dimension and smooth estimation for spectral enhancement are implemented to provide non-redundant dimension-specific information. Furthermore, contrastive learning is employed to align enhanced and source domains for mining compact and evenly separated discriminative features. In addition, a dual-domain balancing loss is built using probability weight, as the nonlearning-based augmentation tends to reinforce the gradient's bias towards imbalance categories. The proposed technique was evaluated on three publicly available datasets, showing comparable generalization performance to advanced multimodal, domain adaptation and domain generalization methods.


## Requirements

CUDA Version: 11.7

torch: 2.0.0

Python: 3.10

## Dataset

The dataset directory should look like this:

```bash
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
└── Pavia
    ├── paviaC.mat
    └── paviaC_7gt.mat
    ├── paviaU.mat
    └── paviaU_7gt.mat

```

## Usage

1.You can download [Houston &amp; Pavia](https://drive.google.com/drive/folders/1No-DNDT9P1HKsM9QKKJJzat8A1ZhVmmz?usp=sharing) dataset here.

2.Run the following command:


Prepare synthetic samples in advance：
```
python Augmentation_H13.py
python Augmentation_PU.py
```

Train on Houston dataset:
```
python train.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --re_ratio 5 --training_sample_ratio 0.8 --flip_augmentation --radiation_augmentation
```
Train on Pavia dataset:
```
python train.py --data_path ./datasets/Pavia/ --source_name paviaU --target_name paviaC --re_ratio 1 
```

