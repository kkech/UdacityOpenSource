# Udacity Project Showcase Challenge 2019

### Title of the Project

Deep learning for detection different categories of chest disease by X-Ray images.

## 1- Project Description

<div align="justify">From what I've learned from the Deep learning is that it can be used in several areas, such as: Object Segmentation, Instance segmentation, Object Detection, Image Classification, Image Classification With Localization, Image Style Transfer, Image Colorization, Image Reconstruction, Image Super-Resolution, Image Synthesis, etc. For each area, there are several Neural network model. We can use either existing models or pre-trained models or we can make our own model, well, if we know how it works. But I noticed that when I started using existing models, I quickly arrived at getting for example "RuntimeError: CUDA out of memory. Tried to allocate --.00 MiB (GPU 0; 2.00 GiB total capacity; -.-- GiB already allocated; --.-- MiB free; --.-- MiB cached)". The problem is that during training the model, it is important to have a good PC with a lot of memory or for example use colab,... So, I tried to understand how the models work and see what parameters are using a lot of memory. Then I tried to make a simple model and applied it to detect different categories of chest disease by X-Ray images.

### Objectif

The objectifs of my project are:

* Try to make models that can work on very modest PCs that include gpu with little memory, because many people have PCs that are not powerful and that also do not have always access to internet and therefore cannot use colab or other.

* Try to create a model that can unify several fields of application, such as **Segmentation, Classification and Object detection**.

* To try in my spare time to realize an application in the medical field to specialize in the identification and analysis of all common categories of chest disease seen by X-Ray images that can help doctors to detect quickly the diseases. The first two points will be included in this application.
* Take into account a secure of the data by using encrypted computation and Differential Privacy.

###  Progress report

For the moment I have focused only on the first part, namely making a simple model that uses little memory during learning. I was able to use **Dedicated GPU memory 0.5/2.0 Go**. So the goal of this code is to design a net model (I named : MyNetModel) that used very small part of the dedicated GPU memory.

The next step will be to try to test on :<p><p>

1. The version 3 of Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images, founded here:
https://data.mendeley.com/datasets/rscbjbr9sj/3  

2. Then I hope use NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories (1, Atelectasis; 2, Cardiomegaly; 3, Effusion; 4, Infiltration; 5, Mass; 6, Nodule; 7, Pneumonia; 8, Pneumothorax; 9, Consolidation; 10, Edema; 11, Emphysema; 12, Fibrosis; 13, Pleural_Thickening; 14 Hernia), from National Institutes of Health - Clinical Center, founded here:  
http://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

3. Next I will try to improve my model to arrive at 99% of accuracy.

4. After that, I'll try to develop MyNetModel to be able to do <b>Semantic Segmentation, Classification and Object detection</b> at the same time, by always taking into account the minimization of the dedicated GPU memory.

5. At the end I will try to secure the X-Ray images data of the patients by using encrypted computation, and use Differential Privacy for deep learning to update the training of deep learning by using the new X-Ray images data when it is possible depending on the powerfull of the PC. Moreover I will take into account the privacy and the secure of the X-Ray images data when a patient want to access to his data by using encrypted computation.

## 2- Requirements

    - torch, torchvision and torchsummary
    - PIL, matplotlib and seaborn
    - numpy and pandas

#### Contributor

|Name                      |   Slack    |
|--------------------------|------------|
| BENHABIB Mohamed Choukri |     BMC    |
