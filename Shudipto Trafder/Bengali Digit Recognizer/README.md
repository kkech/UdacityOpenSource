# Project: Bengali Digit Recognizer
A web app where user can draw Bengali digit and the AI model can detect handwritten digit and predict the digit.

# Schreenshorts
| First Image  | Second Image |
|---| ---|
|  ![First Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn1.png)  | ![Second Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn2.png) |

## Test
Web App:  [Bengali Digit Recognizer](https://bengali-digit-recognizer.herokuapp.com/)

## Framework and Datasets
**Framework**: Pytorch

**Datasets**: [NumtaDB: Bengali Handwritten Digits](https://www.kaggle.com/BengaliAI/numta)

# Architecture
Custom CNN model is used

| CNN Layer | Linear Layer|
|---| ---|
| Layers: 7 | Layers: 2 |
| Normalization: BatchNorm2d | |
| Pooling: MaxPool2d | |
| Activation Function: ReLU | Activation Function: ReLU, Softmax |
| Dropout: 0.25 | Dropout: 0.4|


### Hyper Parameters:
| Hyper Parameters|
|---
| Epoch: 25
| Batch Size: 128
| Learning Rate: 1e-3
| Loss function: NLLLoss
| Optimizer: Adam
| Scheduler: StepLR
| Transformations: Resize(180), RandomRotation(30),
| Data Split: 20%

# Accuracy
Test Accuracy: 99.2890%

| Class wise Accuracy
| ---
| Test Accuracy of     0: 99% (1104/1111)
| Test Accuracy of     1: 99% (1095/1105)
| Test Accuracy of     2: 99% (1094/1099)
| Test Accuracy of     3: 98% (1080/1091)
| Test Accuracy of     4: 99% (1078/1087)
| Test Accuracy of     5: 99% (1088/1096)
| Test Accuracy of     6: 99% (1057/1067)
| Test Accuracy of     7: 99% (1050/1053)
| Test Accuracy of     8: 99% (1124/1127)
| Test Accuracy of     9: 99% (1051/1059)
| Test Accuracy (Overall): 99% (10821/10895)

## Others Libraries
- Pandas
- Numpy
- PIL
- Matplotlib

## Kernel Link
[BN digit with pytorch](https://www.kaggle.com/iamsdt/bn-digit-with-pytorch)

## Deployement
The web app is deployed in *Heroku*

Web App:  [Bengali Digit Recognizer](https://bengali-digit-recognizer.herokuapp.com/)


# Schreenshorts

| First Image  | Second Image |
|---| ---|
|  ![First Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn1.png)  | ![Second Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn2.png) |

## Credits:
I have no experience in web design, so I use this website template from
repo [How to deploy a keras model to production](https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production)
Author: **Siraj Raval**

# Develpoer
**Shudipto Trafder**

**Slack Name:** @Shudipto Trafder

Email: [Shudiptotrafder@gmail.com](mailto:shudiptotrafder@gmail.com)

Linkedin: [Shudipto Trafder](https://www.linkedin.com/in/iamsdt/)