# Federated Learning of U-Net for Pneumothorax Segmentation

## Introduction

This project demonstrates the usage of the PySyft library for the federated learning of PyTorch models on the example of federated pneumothorax segmentation of lung X-ray images with a U-Net model. This is a project I've built during the [Secure and Private AI Scholarship Challenge](https://www.udacity.com/facebook-AI-scholarship) with Udacity. 


## Data 

The data I've used comes from [this Kaggle competition](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation). 

For the demo purposes, I've included the sample images provided by Kaggle (which are obviously not enough to obtain any useful results), keeping in mind that the goal of this repo is not to train a meaningful model, but to merely show how the training of a semantic segmentation model can be done across multiple clients (i.e. hospitals) without compromising the patients' data privacy.

The full Pneumothorax competition dataset needs to be downloaded from Google Cloud using [this tutorial](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/siim-cloud-healthcare-api-tutorial/).


## Installation 

In order to run the code provided in this repo, you need Python 3.6+ and all the packages from the `requirements.txt` file, which can be installed using Pip as shown below:

```
pip install -r requirements.txt
```

## Usage

* The notebook [unet_pytorch_federated_v1.ipynb](./unet_pytorch_federated_v1.ipynb) shows an example of federated learning without parameter aggregation, i.e., by sending the model to a worker, training locally on a batch of data, retrieving it back and then repeating the process with the other batches located on the same or other workers.


* The notebook [unet_pytorch_federated_v2.ipynb](./unet_pytorch_federated_v2.ipynb) demonstrates how the same model can be trained by learning new parameters on each worker separately, and securely aggregating them at the end of each epoch using additive secret sharing. 


## References

* PySyft https://github.com/OpenMined/PySyft
* U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/abs/1505.04597
* PyTorch-Unet https://github.com/usuyama/pytorch-unet 
* Pneumothorax Kaggle competition: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
* Udacity Secure and Private AI (free course) https://www.udacity.com/course/secure-and-private-ai--ud185