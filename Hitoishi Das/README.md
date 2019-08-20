# Medical sIRNA Detection

A very heavy DL model, which detects levels of small Interfacing RNA (sIRNA) on a strand of a cell batch and by that, we can calculate the efficiency of a drug on a disease using AI

## Desired Outputs

Since I do not have a proper hardware system for a dataset worth 46GB, and Kaggle kernels have a limitation of only 9 hours running time, my training loss came down to as high as 3.2, although with 25-30 more iterations, it would come down to around 1.

## Details

The framework used is PyTorch. I have gone for a DenseNet-201 pretrained model. The optimizer is an Adam optimizer and with a constant learning rate of 0.003


## Data Source
The data can be found at https://www.kaggle.com/c/recursion-cellular-image-classification/data


