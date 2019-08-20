# Federated Blindness Detection

Contributors: Labiba Kanij Rupty (@Labiba), Kris Akira Stern (@K.S.)

## Aim
We aim to apply federated learning to some datasets originated from the APTOS 2019 Blindness Competition. This would be an apt application because for such a dataset containing potential sensitive patient informaiton, federated learning would be ideal to protect client privacy. 

## How it will benefit people?
This project is a proof-of-concept. But it can be extended to real life situation where the sensitive fundus photographs / retina images is to be stored in a remote server, and the aim is to train the existing model (called "APTOSNet") while brining the model to the data, and not the other way around. The implications of such an application in user-facing situations are many, including the possibilty or more healthcare institutions releasing their protected patient records to the wider scientific community for research purposes, while retaining client privacy. 

## What material from the course has been used?
We have applied concepts and techniques from deep learning, and have applied it in our case to process a unique data set that is small in number of images contained though with at least 5 classifications. Other computer vision techniques have been applied as well, including some tricks we have incorporated for data pre-processing in order to remove the unwanted dark pixels that normally form the rim of the retina image. 

## Why the particular APTOSNet model used in the project?
The so-called APTOSNet project adopted by us has been designed to be simple yet fast. It is composed of four initial convolutional layers, then three fully connected (FC) layers and two dropout layers in-between. We aim for a light weight CNN model that is specifically designed for the federated training, in order for the code to produce results with good efficiency, though probably at a cost of test accuracy. The current test accuracy of our model is at most around 71.2% without transfer learning. With transfer learning using ResNet, this metric can be pushed up to at least 80%.

## How the model can be run?
The code on the notebook can be run on Google Colab connected to some local runtime. Once the source APTOS datasets (https://www.kaggle.com/c/aptos2019-blindness-detection/data) have been downloaded and put in an accessible location, the code can be run. 

## How can the code be improved?
Transfer learning is a technique that can definitely be used to improve the test accuracy. Other possible avenues to explore which could potentially improve the accuracy of our work are siamese networks and one-shot learning. We will strive to improve our work and implement the different options available to obtain a great prediction accuracy. 

## Resources
1. [An_enhanced_diabetic_retinopathy_detection_classification](https://github.com/rupaai/APTOS_blindness_detection/blob/master/Resources/An_enhanced_diabetic_retinopathy_detection_classification.pdf)
2. [Importance of data pre-processing](https://github.com/rupaai/APTOS_blindness_detection/blob/master/Resources/CentreCropping.png)
3. [Convolutional Neural Networks for Diabetic Retinopathy](https://github.com/rupaai/APTOS_blindness_detection/blob/master/Resources/Convolutional_Neural_Networks_for_Diabetic_Retinopathy.pdf)
4. [Diabetic Retinopathy Detection Using Machine](https://github.com/rupaai/APTOS_blindness_detection/blob/master/Resources/Diabetic_Retinopathy_Detection_Using_Machine.pdf)
5. [Diabetic Retinopathy Detection Using Deep Convolutional Neural Networks](https://github.com/rupaai/APTOS_blindness_detection/blob/master/Resources/Diabetic_Retinopathy_Detection_using_Deep_Convolutional_Neural_Network.pdf)
6. [Fundas Image Classification Using VGG](https://github.com/rupaai/APTOS_blindness_detection/blob/master/Resources/Mateen_2019_Fundus_Image_Classification_Using_VGG_19_MDPI.pdf)
