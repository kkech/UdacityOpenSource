# SPAIC | SG_AIMLPros

SPAIC - 2019 Udacity Facebook Secure & Private AI Challenge Scholarship Program  

Portfolio | Website - https://drvigneshwari.github.io/SG_AIMLPros-MOST

# Team MOST Project Showcase Challenge

Project <b>MOST</b>, as is suggested by the name, focuses on <b>MO</b>deling and <b>ST</b>atistics.  Its mission is to implement different models on the same dataset and provide statistics regarding the most accurate model for image processing through details collected from the models.

The goal of this project is to classify rice diseases using a rice disease dataset on Kaggle. By using various pretrained models such as VGG19 and ResNet50, we try to determine the best model for this task and analyze the pros and cons of each model, with the end goal of competing in the Udacity Facebook Secure & Private AI Scholarship Project Showcase Challenge.


# Project Overview:

Rice is the staple food for more than half the world's population; accordingly, its supply must double by 2050 to keep up with food demand from population growth. Nearly a fifth of the world’s population are rice farmers. However, they lose approximately 37% of their rice crop yields to various rice diseases. 


<p align="center" xmlns="http://www.w3.org/1999/html">
  <img src="https://davidcliveprice.com/wp-content/uploads/2014/11/asian-rice-farmers.jpg" width="350" title="hover text">
<br/>
Image Credit: <a href="https://davidcliveprice.com/wp-content/uploads/2014/11/asian-rice-farmers.jpg">Davidcliveprice.com</a>
</p>
  

<p>We came up with a model that classifies whether a rice leaf </p>


| Types1 | Types2 | Types 3 | Types 4
| --- | ---| ---| ---
| Healthy | Brown spots | Leaf blast| Hispa
| <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRVdIquvBXmAJo2M1YfCDbJajiaB1BAcOMxJYT0qr82HOsRAIcL" width="150" height="200" title="hover text">|  <img src="http://www.knowledgebank.irri.org/images/stories/brown-spot-1.jpg" width="150" height="200" title="hover text">| <img src="https://miro.medium.com/max/706/1*5OIVPVgxZyPy-OE2Lw8lBg.png" width="150" height="200" title="hover text">| <img src="https://content.peat-cloud.com/hd/rice-hispa--1.jpg" width="150" height="200" title="hover text">|

<b><p>We believe that our classification model will help rice farmers better understand the condition of their rice crops and in turn lead to higher yields.
</p></b>

<i><p> For a more detailed project introduction: https://github.com/risper25/MOST- (@risper bevalyn)</p></i>


# Project Participants
 
Following are the Slack handles of the members (total count: 7)

| Name| Slack Name| Github
|--- | ---| --- |
| Vigneshwari Ramakrishnan|@Vigneshwari|https://github.com/drvigneshwari
| Shudipto Trafder |@Shudipto Trafder | https://github.com/Iamsdt 
| Laura Truncellito |@LauraT | https://github.com/ltruncel
| Ellyana Linden |@Ellyana Linden | https://github.com/ellyanalinden
|Risper Bevalyn Adera |@risper bevalyn | https://github.com/risper25
| Shuvam Lal | @happycoder354 |https://github.com/shuvamlal9
| Anna Scott | @Anna Scott | https://github.com/forfireonly

# Contributions made by the team

## 1) Pre-trained model implementation
1) @Ellyana Linden
2) @Shudipto Trafder
3) @Vigneshwari 
4) @risper bevalyn
5) @happycoder354
6) @Anna Scott

## 2) Technical documentation | Other
|Slack Name|Contribution Areas|
|---| ---|
| @LauraT |Description of the models, Readme file |
| @Vigneshwari |Readme file, project deck, Github Project repo, portfolio |
| @risper bevalyn| Introduction of the project, data collection and chart in google sheet |
| @Shudipto Trafder| Essential attributes identification and listing, Readme file |


# Dataset utilized

Datasets: [Rice Diseases Image Dataset](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset)

## Details of the dataset

Dataset of rice images -  4 labels/categories (3 diseases and pests, 1 healthy) 
1) Brown spots (523 files)
2) Healthy (1488 files)
3) Hispa (565 files)
4) Leaf blast (779 files)


# Pretrained models used in this project

1) GoogleNet
2) ResNet50
3) VGG19
4) Resnext101_32x8d
5) Densenet201
6) Alexnet
7) VGG16

## Link to our project deck / activity records

Link: https://github.com/drvigneshwari/SG_AIMLPros-MOST/projects/1

## Links to all models implemented by the participants 

| Slack Name| Model Name | Kernel Link |
| --- | ---| ---|
| @Ellyana Linden | GoogleNet | [googlenet.ipynb](https://github.com/ellyanalinden/rice_diseases_kaggle/blob/master/googlenet.ipynb)
| @Ellyana Linden | Resnet50 | [googlenet](https://www.kaggle.com/ellyanalinden/googlenet)
| @Shudipto Trafder | resnext101_32x8d | [kernelcd66c7e01d-version1](https://www.kaggle.com/iamsdt/kernelcd66c7e01d?scriptVersionId=17795486)
| @Shudipto Trafder | Resnet50 | [kernelcd66c7e01d](https://www.kaggle.com/iamsdt/kernelcd66c7e01d)
| @Vigneshwari | Alexnet | [kernel4946364e36](https://www.kaggle.com/drvigneshwari/kernel4946364e36)
| @risper bevalyn, @happycoder354 | VGG19 | [rice-leaves-disease-classifier](https://www.kaggle.com/risperbevalyn/rice-leaves-disease-classifier?scriptVersionId=17832475)
| @Anna Scott | VGG16 |[kernel-vgg16](https://www.kaggle.com/scottanna/kernel-vgg16/)


# Implementation and Analysis

### Test accuracy:
GoogleNet results had the highest accuracy (97%), followed by ResNext101_32x8 (81%), VGG16 (81%), ResNet50 (79%, 10 epochs), ResNet50 (69%, 25 epochs), Alexnet (66%), and VGG19 (61%)

### Optimizers:
Two optimizers were selected: Adaptive Moment Estimation (Adam) and Stochastic Gradient Descent (SGD).  Adam is used by @Shudipto Trafder for training resnet 50, and the remaining models were all trained with SGD as an optimizer.

### Loss function:
Cross-entropy loss, or log loss, is used to measure the performance of all classification models in this project.

### Learning rate:
The learning rates of pretrained models selected in this project range between 0.001 and 0.003.  

### Results from analysis:

![2019-08-20 11_57_29-riceproject_details - Google Sheets](https://user-images.githubusercontent.com/50787118/63322739-c06a6700-c341-11e9-9bdb-49223630b24b.png)
 
![2019-08-20 11_57_22-riceproject_details - Google Sheets](https://user-images.githubusercontent.com/50787118/63322741-c102fd80-c341-11e9-8104-c7628b149a34.png)

<p>For details of the test accuracy and epochs:</p>
<p>https://docs.google.com/spreadsheets/d/1ZMs9j4nvNw2387g6x5ZdN9uEodQqYqnViGnDU-fwNxQ/edit#gid=0</p>



# Conclusion

GoogleNet is a class of architecture designed by researchers at Google. It was the winner of ImageNet 2014, and it is proven to be a winner for our rice disease project as well. Not only its architecture goes deeper, researchers also took a noval approach in its structure, the so called inception model:

<br>
<p align="center">
  <img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/08131905/temp10.png" width="400" height="300" title="hover text">
  </p>
</br>

As we can see, multiple types of feature extractors are present, which is drastically different from the sequential architectures of other models.  Also, the final architecture contains multiple inception modules stacked one over the other, which helps the model converges faster.  

Through implementation of the rice project, we have also found that GoogleNet is superior than AlexNet, Resnet50, Resnext101_32x8, or VGG19 for the purpose detecting rice diseases.

# Value behind this project

This project is the implementation of advanced Machine learning techiques which can create a great impact in the life of farmers and all the life existense in this world. The primary utilization of this project may greatly impact in the <b>domain of agriculture.</b> 

According to the research by Kihoro, J., Bosco, N.J., Murage, H. et al. SpringerPlus (2013) [2: 308. https://doi.org/10.1186/2193-1801-2-308], Investigating the impact of rice blast disease on the livelihood of the local farmers in greater Mwea region of Kenyamers, conveyed that <b>rice blast disease is the most destructive disease compared to other diseases the farmers mentioned that it is possible to harvest nothing when affected by the disease. </b>

### Percentage rice blast incidences in the various planting group from 2006 to 2010.
![2019-08-20 11_41_36-Investigating the impact of rice blast disease on the livelihood of the local fa](https://user-images.githubusercontent.com/50787118/63321822-813b1680-c33f-11e9-9e81-9bd54699661f.png)

[Image credit: https://link.springer.com/article/10.1186/2193-1801-2-308#copyrightInformation]

### Figure shows the rice blast disease cases created and pointed with point symbol on the Mwea region map
![2019-08-20 11_39_24-Investigating the impact of rice blast disease on the livelihood of the local fa](https://user-images.githubusercontent.com/50787118/63321742-3ae5b780-c33f-11e9-8e8c-c30e7087728c.png)

[Image credit: https://link.springer.com/article/10.1186/2193-1801-2-308#copyrightInformation]

From this analysis, it is <b> predominantly highlighting the loss faced by the farmers in the Mwea region of Kenyamers. This loss is not limited to farmers of Kenyamers but also throughout the world.</b> 

Hence, taking this problem statement forward we are trying to provide a better solution to identify the rice diseases using the best model out of other pre-trained models for image classification and have achieved the goal of this project with the help of the learnings gained through Udacity Facebook AI Secure and Privacy Scholarship Challenge of 2019.    

# Taking the project further

This is a baselined project tested and identified best model to be used for the classification. To take this project forward, we can create a mobile application to detect the rice diseases at instant, with better user-interface enabling the usage at ease for any kind of users. 

This mobile application can utilize the approch of Federated learning enabling the application in mobile phones to learn a shared prediction model by collaborating with each other (other mobile devices) while keeping all the training data on device, decoupling the ability to do machine learning from the need to store the data in the cloud. This goes beyond the use of local models that make predictions on mobile devices (like the Mobile Vision API and On-Device Smart Reply) by bringing model training to the device as well.

For more details on Federated learning: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

![2019-08-20 13_27_05-Google AI Blog_ Federated Learning_ Collaborative Machine Learning without Centr](https://user-images.githubusercontent.com/50787118/63329023-d0d50e80-c34e-11e9-8865-6937a8aa3ea1.png)

[Image credit: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html]

This approach with the combination of 

<i>federated learning</i></b> and the <b><i>best model</i></b> identified through this project, helps the model refined with accuracy day-by-day in identification of the rice diseases in-addition ensuring the privacy of the user's data too. 

# Learnings implemented from SPAIC
Udacity Facebook AI Secure and Privacy Scholarship Challenge of 2019
<ul> 
 <li>Federated learning</li>
 <li>Model | Train | Test | Deploy datasets using pre-trained model of image classification</li>
 <li>Increasing the accuracy of the model using epochs and other approaches</li>
 <li>Hyperparameter Tuning</li>
 <li>Implementation | Usage of PyTorch</li>
 <li>Utilization of different model optimizers, loss functions, learning rate, freezing layers etc</li>
</ul>



# References
@Ellyana Linden - DLND Udacity course

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

https://pytorch.org/docs/stable/torchvision/models.html

@Vigneshwari

https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

https://pytorch.org/docs/stable/torchvision/models.html

@LauraT

<p>CNN Architectures: LeNet, AlexNet, VGG, GoogLeNet, ResNet and more</p> 
<p>https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5</p>

<p>Going Deeper with Convolutions</p>
<p>https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf</p>

<p>10 Advanced Deep Learning Architectures Data Scientists Should Know</p>
<p>https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/</p>

<p>TORCHVISION.MODELS</p>
<p>https://pytorch.org/docs/stable/torchvision/models.html</p>

@happycoder354 

https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624
 

