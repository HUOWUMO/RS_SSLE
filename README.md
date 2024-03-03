# Invariance-Driven Non-parametric Covariate Expansion for Cross-Scene Hyperspectral Image Classification

<p align='center'>
  <img src='abstract_00.png' width="800px">
</p>

## Abstract

For hyperspectral cross-scene classification tasks, where the target domain is invisible, the model can only be trained in the source domain (SD) and then applied directly to the target domain (TD). Diverse atmospheric conditions, lighting schemes, and imaging equipment across various scenes contribute to variations in the spectral reflection, which constitute the primary obstacle to this domain generalization problem. Current researchs have demonstrated that the convolutional neural network (CNN) can effectively address the disparity between SD and TD when combined with the learnable data augmentation network. Given this performance, one may inquire whether domain expansion can be achieved via heuristic data augmentation to improve CNN's generalization capabilities. Drawing inspiration from this, the article presents a non-parametric approach to generate a hyperspectral "sketch style" for spectral-spatial augmentation. The method consists of weighted resampling in space and developing a model to generate a downsampling approximation of spectral data. Supervised contrastive learning is also utilized to refine the feature extraction network so that the feature representation of the real and virtual data are nearly invariant. In light of the fact that this enhancement method preserves the original label distribution, we employ and simplify the standard focal loss as a regularization term for inter-class gradient balancing. Comprehensive experiments conducted on three publicly available datasets demonstrate that the proposed method are on par with those of multimodal and advanced domain adaptation/generalization approaches.


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

