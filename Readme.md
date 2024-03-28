# Accelerating Neural Network Training: A Brief Review

## Overview of the repo
This repository contains the official code for the research paper titled "Accelerating Neural Network Training: A Brief Review" available at https://arxiv.org/pdf/2312.10024.pdf. The research employs sophisticated methodologies, including Gradient Accumulation (GA), Automatic Mixed Precision (AMP), and Pin Memory (PM), to optimize performance and expedite the training process. The study compares the performance of various deep learning models such as EfficientNet-BO, ResNet50, and Pretrained VIT in image classification tasks using PyTorch with/without employing those performance techniques. The experiments have been conducted using well-known CIFAR-10 and CIFAR-100 image datasets.

## Abstract
The process of training a deep neural network is characterized by significant time requirements and associated costs. Although
researchers have made considerable progress in this area, further work is still required due to resource constraints. This study
examines innovative approaches to expedite the training process of deep neural networks (DNN), with specific emphasis on
three state-of-the-art models such as ResNet50, Vision Transformer (ViT), and EfficientNet. The research utilizes sophisticated
methodologies, including Gradient Accumulation (GA), Automatic Mixed Precision (AMP), and Pin Memory (PM), in order to
optimize performance and accelerate the training procedure.

The study examines the effects of these methodologies on the DNN models discussed earlier, assessing their efficacy with
regard to training rate and computational efficacy. The study showcases the efficacy of including GA as a strategic approach,
resulting in a noteworthy decrease in the duration required for training. This enables the models to converge at a faster pace.
The utilization of AMP enhances the speed of computations by taking advantage of the advantages offered by lower precision
arithmetic while maintaining the correctness of the model.

Furthermore, this study investigates the application of Pin Memory as a strategy to enhance the efficiency of data transmission
between the central processing unit and the graphics processing unit, thereby offering a promising opportunity for enhancing
overall performance. The experimental findings demonstrate that the combination of these sophisticated methodologies significantly
accelerates the training of DNNs, offering vital insights for experts seeking to improve the effectiveness of deep learning processes.

## Getting Started
### Requirements 

- Python3
- Pytorch (>1.0)
- torchvision (>0.2)
- numpy
- pillow~=6.2.1
- torch_optimizer
- randaugment
- easydict
- pandas~=1.1.3

## Run the code
- Each model code can be directly run on the GPU. Change the runtime type to GPU if running in Google Colab.

## Models
- EfficientNet-BO: EfficientNet-BO is a family of convolutional neural network architectures designed to achieve state-of-the-art performance while being computationally efficient. These models are characterized by a novel compound scaling method that uniformly scales all dimensions of depth, width, and resolution. This scaling approach enables EfficientNet models to achieve remarkable accuracy across a wide range of tasks while maintaining a smaller number of parameters compared to other architectures.

- ResNet50: ResNet50 is a variant of the ResNet (Residual Network) architecture, which is renowned for its deep neural network design featuring skip connections or residual connections. ResNet50 specifically consists of 50 layers and has been widely used for various computer vision tasks, including image classification, object detection, and semantic segmentation. Its skip connections help alleviate the vanishing gradient problem, enabling training of very deep networks effectively.

- Pretrained VIT (Hugging face): The Pretrained Vision Transformer (VIT) model, available through Hugging Face's Transformers library, is based on the Vision Transformer architecture. VIT represents images as sequences of patches and processes them using Transformer layers, originally designed for natural language processing tasks. By leveraging pretraining on large-scale datasets like ImageNet, Pretrained VIT models learn rich visual representations that can be fine-tuned for downstream tasks such as image classification, object detection, and image segmentation, achieving competitive performance compared to traditional convolutional neural networks.

## Datasets
## CIFAR-10 and CIFAR-100
The CIFAR-10 and CIFAR-100 datasets are commonly used benchmark datasets in the field of computer vision. They consist of 60,000 32x32 color images in 10 and 100 classes respectively, with 6,000 images per class. These datasets are widely used for training and evaluating image classification models.

### CIFAR-10:
- CIFAR-10 is composed of 10 classes, each representing a different type of object, such as airplanes, automobiles, birds, cats, etc. It is often used for quick experiments and initial model testing due to its smaller size compared to CIFAR-100.

### CIFAR-100:
- CIFAR-100 is an extension of CIFAR-10, with 100 classes instead of 10. Each class contains 600 images. It is more challenging than CIFAR-10 due to the larger number of classes, making it suitable for evaluating the robustness and generalization capabilities of models.

Both datasets are preprocessed and split into training and test sets, allowing for consistent evaluation of model performance across different architectures and methodologies.

## Metrics Evaluated
Two key metrics are evaluated to assess the performance of the models:

- **Accuracy**: This metric measures the proportion of correctly classified instances out of all instances evaluated. In the context of image classification, it represents the percentage of images correctly classified by the model.

- **F1 Score**: The F1 score is the harmonic mean of precision and recall. It considers both the precision (the number of true positive predictions divided by the total number of positive predictions) and recall (the number of true positive predictions divided by the total number of actual positives) to provide a single score that balances both measures. It is particularly useful when dealing with imbalanced datasets, as it takes into account both false positives and false negatives.

## Loss Function
Cross Entropy is a commonly used loss function in classification tasks, including image classification. It measures the difference between the predicted probability distribution and the actual probability distribution of the classes. It is particularly effective when dealing with multi-class classification problems, such as CIFAR-10 and CIFAR-100, as it penalizes incorrect classifications more heavily, leading to better model training and convergence.


## Performance Tuning Techniques Leveraged
- Automatic Precision Scaling: This technique dynamically adjusts the numerical precision of computations during training, allowing for more efficient use of computational resources while maintaining or improving model accuracy. It enables deep learning models to leverage lower precision arithmetic when feasible, reducing memory consumption and accelerating training speed without sacrificing performance.

- Pin Memory: Pin Memory is a memory management technique commonly used in deep learning frameworks like PyTorch. It involves pinning memory allocations in the system's physical memory, which allows for faster data transfer between CPU and GPU during training. By preventing memory from being swapped out to disk, Pin Memory reduces overhead and latency, improving overall training performance.

- Gradient Accumulation: Gradient Accumulation involves accumulating gradients over multiple batches before updating the model parameters. This technique is particularly useful when dealing with large batch sizes that may not fit into GPU memory. Instead of updating the model after processing each batch, gradients are accumulated over several batches, and the model parameters are updated less frequently. This can lead to more stable training, especially in scenarios where GPU memory is limited or when training on large datasets.


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

## Citation 
If you find this work useful, please consider citing the following paper:

```angular2
@misc{nokhwal2023accelerating,
      title={Accelerating Neural Network Training: A Brief Review}, 
      author={Sahil Nokhwal and Priyanka Chilakalapudi and Preeti Donekal and Suman Nokhwal and Saurabh Pahune and Ankit Chaudhary},
      year={2023},
      eprint={2312.10024},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```




