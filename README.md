# Waste Classification using CNNs and Transfer Learning

## Overview

This project implements an end-to-end deep learning pipeline for multi-class image classification of waste materials. The objective is to classify images into categories such as cardboard, glass, metal, paper, plastic, and trash using both a custom Convolutional Neural Network (CNN) and a transfer learning approach.

The project focuses not only on model building but also on **systematic debugging, performance optimization, and generalization improvement**.



 
## Objectives

* Develop a custom CNN for image classification from scratch
* Implement transfer learning using MobileNetV2
* Improve generalization through regularization and augmentation
* Diagnose and resolve overfitting and training instability
* Compare model performance and extract actionable insights




## Dataset

🔗 Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

### Description

* ~2500 images across 6 classes:

  * Cardboard
  * Glass
  * Metal
  * Paper
  * Plastic
  * Trash

### Expected Directory Structure

```
Garbage classification/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```




## Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Matplotlib
* Scikit-learn
* Google Colab




## Data Pipeline



### Preprocessing

* Images resized to 224×224
* Two preprocessing strategies:

  * Custom CNN → pixel normalization (1/255)
  * Transfer Learning → MobileNetV2 `preprocess_input`
* Train-validation split: 80/20

### Data Augmentation

To mitigate overfitting and improve robustness:

* Random horizontal flip
* Random rotation
* Random zoom
* Random contrast adjustment

### Performance Optimization

* `tf.data` pipeline with caching, shuffling, and prefetching
* Efficient batch loading for GPU utilization





## Model 1: Custom CNN

### Architecture

```
Conv2D(32) → ReLU → MaxPool
Conv2D(64) → ReLU → MaxPool
Conv2D(128) → ReLU → MaxPool
Flatten
Dense(128) → ReLU
Dropout(0.5)
Dense(6) → Softmax
```


### Training Strategy

* Loss: Sparse Categorical Crossentropy
* Optimizer: Adam
* EarlyStopping on validation loss

### Performance

* Training Accuracy: ~68%
* Validation Accuracy: ~64%

### Observations

* Initial severe overfitting (~99% training accuracy)
* Improved using:

  * Dropout
  * Data augmentation
  * Early stopping





## Model 2: Transfer Learning


### Base Model

MobileNetV2 pretrained on ImageNet

### Approach

* Removed top classification layers
* Froze base model for feature extraction
* Added custom classification head
* Applied fine-tuning on top layers

### Architecture

```
Data Augmentation
MobileNetV2 (frozen)
GlobalAveragePooling
BatchNormalization
Dense(128) → ReLU
Dropout(0.5)
Dense(6) → Softmax
```

### Training Strategy

* Reduced learning rate for stability
* Fine-tuning with very low LR (1e-5)
* EarlyStopping

### Performance

* Validation Accuracy: ~71%

### Observations

* Significant improvement over custom CNN
* Demonstrates effectiveness of pretrained feature extraction





## Challenges & Solutions

### 1. Overfitting

**Problem:**

* Training accuracy ~99%, poor validation performance

**Solution:**

* Introduced dropout layers
* Applied aggressive data augmentation
* Used EarlyStopping





### 2. Transfer Learning Instability

**Problem:**

* Model stuck at ~20% accuracy

**Root Cause:**

* Incorrect preprocessing pipeline
* Incompatible input scaling

**Solution:**

* Replaced normalization with MobileNetV2 `preprocess_input`
* Reduced learning rate
* Properly froze base model





### 3. Dataset Limitations

**Issues:**

* Small dataset size
* High inter-class similarity

**Impact:**

* Limited maximum achievable accuracy
* Increased misclassification between similar classes





## Model Comparison

| Model             | Validation Accuracy | Key Insight                                     |
| ----------------- | ------------------- | ----------------------------------------------- |
| Custom CNN        | ~64%                | Learns from scratch, limited by data            |
| Transfer Learning | ~71%                | Better generalization using pretrained features |





## Error Analysis

* Frequent confusion between:

  * Paper vs Cardboard
  * Plastic vs Trash

**Insight:**
Visual similarity between classes impacts model performance more than architecture complexity.




## Skills Demonstrated

* Deep Learning (CNN architecture design)
* Transfer Learning & Fine-tuning
* Data Pipeline Optimization (`tf.data`)
* Model Regularization Techniques
* Debugging Training Failures
* Performance Analysis & Model Comparison





##  Conclusion

This project highlights the importance of **model selection, preprocessing correctness, and generalization strategies** in deep learning workflows.
It demonstrates a complete ML pipeline — from data ingestion to model optimization with a focus on **practical problem-solving and performance trade-offs** rather than raw accuracy.


