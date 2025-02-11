# Fast Axiomatic Attribution for Neural Networks

## Overview

The **Fast Axiomatic Attribution** paper introduces a novel class of neural networks called **X-DNN (Bias-Free Deep Neural Networks)**. X-DNNs remove the bias terms from all layers of a network, enforcing homogeneity. This leads to improved attribution quality and interpretability, without sacrificing model performance. The use of **Batch Normalization** ensures that the network's accuracy remains comparable to standard networks while enhancing explainability.

### Key Contributions:
- **X-DNN (Bias-Free Deep Neural Networks)**: By removing bias terms from all layers, X-DNNs provide a more faithful attribution while retaining high accuracy.
- **X-Gradient (X-Grad)**: An efficient gradient-based attribution method that identifies input features contributing most to the model’s predictions, offering a computationally efficient alternative to traditional methods like Integrated Gradients.

## Pretrained X-DNN Models

The following pretrained X-DNN models are available for implementing **X-DNN** and **X-Grad** in your projects. I specifically used **X-VGG16** for my project. These models are available from the [Fast Axiomatic Attribution GitHub repository](https://github.com/visinf/fast-axiomatic-attribution):

| Model             | Top-5 Accuracy | Download Link                                   |
|-------------------|----------------|-------------------------------------------------|
| **X-AlexNet**     | 78.54%         | [Download X-AlexNet](xalexnet_model_best.pth.tar) |
| **X-VGG16**       | 90.25%         | [Download X-VGG16](xvgg16_model_best.pth.tar)   |
| **X-ResNet-50**   | 91.12%         | [Download X-ResNet-50](xfixup_resnet50_model_best.pth.tar) |

These models demonstrate that by removing bias terms, **X-DNNs** can achieve comparable performance to traditional models while significantly improving the interpretability and attribution quality of deep learning models.

## File Structure

The repository includes the following key files:

- **`X-grad_with_X-VGG16.py`** / **`X-grad_with_X-VGG16.ipynb`**:
  - This file contains the implementation of **X-Grad** with the pretrained **X-VGG16** model, where the bias terms have been removed (X-DNN). It also computes the saliency maps for model interpretability.
  
- **`Paper4_VGG19.py`** / **`Paper4_VGG19.ipynb`**:
  - This file was created to compare how a pretrained model, initially trained with bias, behaves when the bias terms are removed. It demonstrates the difference in saliency and attribution results when using both **X-DNN** and **X-Grad** and just using **X-Grad**.
  - Both the methods were not even able to predict the correct class labels which means they were learned a lot from the bias out of which X grad plus X DNN performed the worst.


  
