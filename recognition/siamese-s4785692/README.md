# Siamese Network | SIIM-ISIC Melanoma Classification

Author: Sohtaroh Arakawa (s4785692)

## Table of Contents
- [Overview](#overview)
- [Dependences](#dependences)
- [Implementation](#implementation)
- [Results](#retults)
- [Reference](#reference)

## 1. Overview
The objective of this project is to classify medical images of skin lesions into benign or malignant, using a Siamese Network-based classifier. The classifier is trained on ISIC 2020 Kaggle Challenge dataset and aims to achieve 80% accuracy or above on evaludation dataset.

### 1.1 Siamese Network
A Siamese Network consists of two identical subnetwork which generates embeddings for a pair of inputs. By comparing these encoded features, the netwrok measures their similarity and uses this information to perform classification.

The choice of embedding models varies depending on what the model aims to achieve. Residual Network (ResNet) is one of the benchmark when dealing with image classfication. In this project, I selected Vision Transformer (ViT) for twin subnetwork to retrieve a meaningful embedding.


### 1.2 ISIC 2020 Kaggle Challenge Dataset
The dataset used for this project is ISIC 2020 Kaggle Challenge Dataset. For efficient training, images were resized to 224x224, and it consists of 33126 total images. One of the challnge in this dataset is that the class distribution is significantly imbalanced. Due to the characteristic of medical data, the number of samples for malignant images are extremely limited. To address this issue, few techniques were employed which will be described later.

| Label        |  Samples  |
|:-------------|:---------:|
| 0 (Benign)   | 32542     |
| 1 (Malignant)| 584       |


## Dependences
This projected was constructed with the following dependences.
- Python 3.10.19
- PyTorch 2.9.0
- TorchVision 0.24.0
- TorchAudio 2.9.0
- NumPy 2.2.6
- pandas 2.3.3
- Pillow 12.0.0
- scikit-learn 1.7.2 
- tqdm 4.67.1
- PyYAML 6.0.3

### Installation
You can create an environment with required dependences by running the following.
```
conda env create -f environment.yml
```

## Implementation

### Training
- model comparison
- fine tuning process
### Prediction
- evaluation metrics (process and explain what it determines)

## Results
- evaluation metrics

## Reference
G. Koch, R. Zemel, R. Salakhutdinov et al., “Siamese neural networks for one-shot image recognition,” in
ICML deep learning workshop, vol. 2. Lille, 2015, p. 0.




