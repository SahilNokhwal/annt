# Deep Learning Model Performance Evaluation for Image Classification

## Overview
This repository contains code for evaluating the performance of different deep learning models for image classification using PyTorch. The models are evaluated on the CIFAR-10 and CIFAR-100 datasets.

## Prerequisites
This code does not require any prerequisites. Each model code can be directly run on the GPU. Change the runtime type to GPU if running in Google Colab.

## Models
- EfficientNet-BO
- ResNet50
- Pretrained VIT (Hugging face)

## Datasets
- CIFAR-10
- CIFAR-100

## Metrics Evaluated
- Accuracy
- F1 Score

## Loss Function
- Cross Entropy

## Performance Tuning Techniques Leveraged
- Automatic Precision Scaling
- Pin Memory
- Gradient Accumulation

## Performance Tuning Evaluation Scores for CIFAR-10

| Model           | Accuracy | F1 Score | Execution Time |
|-----------------|----------|----------|----------------|
| EfficientNet-BO | 96%      | 96%      | 1 hour         |
| VIT             | 97%      | 97%      | 2 hours        |
| ResNet50        | 92%      | 92%      | 46 minutes     |

## Performance Tuning Evaluation Scores for CIFAR-100

| Model           | Train Accuracy | Test Accuracy | Execution Time |
|-----------------|----------------|---------------|----------------|
| EfficientNet-BO | 81%            | 90%           | 1.1 hours      |
| ResNet50        | -              | 75.2%         | 18 minutes     |
| VIT             | 84%            | -             | 2.5 hours      |

*Note: When train and test accuracy are the same for a model, they are mentioned as 'Accuracy'; otherwise, the difference is mentioned.*

## Performance Tuning Techniques Applied
- Gradient Accumulation: Increase batch size to reduce training execution time.
- Automatic Precision Scaling: Leveraging Autocast and grad scaler to reduce memory load and increase training speed by automatically changing computations from fp32 to fp16 when not required.
- Pin Memory: Set to true in data loader to reduce memory load.

## Code Structure
The code repository contains three directories:
- ResNet50
- ViT
- EfficientNet

Each directory contains self-contained code for the respective model. There are four files in the ResNet50 and EfficientNet directories and two in the ViT directory:
- CIFAR10 and CIFAR100 with performance tuning
- CIFAR10 and CIFAR100 without performance tuning

To run a model with a particular dataset with (or without) performance tuning, simply run the respective `.py` or `.ipynb` file.

*Note: For running VIT code without performance tuning, the trainer arguments for gradient accumulation and fp16 parameter are to be commented to get the execution time statistics for the default batch size of 70. The execution time screenshots of the model runs are uploaded in the VIT folder.*

