# Project - MOBILE APPLICATION TO DETECT AND COUNT MONEY TO ASSIST THE VISUALLY IMPAIRED (INCOMPLETE)

## Description

- Money count, Is a proposed Mobile Application that utilizes the camera of a mobile device to scan images of the Nigerian Naira currency and can also be extended to other currencies, in order to recognize, classify and then give audio feedback to the user before summing up the denominations. This is to use the power of Deep learning to give Blind and Visually Impaired people in Nigeria the opportunity to count money by themselves in order to address the issue in Nigeria where people living with visual Impairment get scammed and swindled by other people when asking for assistance to use the ATM, count money or shopping as a result of their disability. 

-  I used the ResNet architecture because of its relatively small size and good accuracy. I modified the ResNet architecture by changing the final sequential layer to classify the banknotes correctly. I made use of PyTorch to create and train the model and then I converted the gradients of the trained model to the ONNX and PB format in order to easily integrate into a mobile application.

- In order to obtain the training and test dataset I used in training the classifier and validating, I utilized 3  main resources. I took pictures of the different denominations of the Naira note by myself, I collected images from people and I also downloaded images from the internet.

-  In order to offer privacy protection to the Individuals whose data I used, I made use of three privacy protection mechanisms. I anonymized the dataset by changing the names of all the images and replacing them with indexes after that I also applied Local Differential Privacy by randomly mixing my Images and also the Images got from the Internet with the Images I got from friends to protect against Differencing Attacks. Also, the validation of the currencies the user scans from the camera would take place offline on the devices, to ensure that no internet connection is needed and also that no image is uploaded to the cloud. This protects privacy and ensures protection from cyberattacks and hacking. Also, Application updates would be gotten by users by receiving updated gradients without any data exchange between the Developers and the users.

-  As a result of the limited number of Datasets, I performed different data augmentation techniques such as Horizontal flipping, Vertical flipping, cropping e.t.c. In order to augment the dataset to improve learning.

- As of right now, due to the Time constraint, lack of expertise in some key areas and other factors, I could only implement the Naira currency classifier and make a wireframe for the application UI, but the idea can be extended to other currencies and the application can be implemented later.

- Due to the large size of the train and test dataset, The datasets are not uploaded on Github but instead can be downloaded from the google drive link below. [Google Drive link for the datasets](https://drive.google.com/open?id=1XuJezzMLFHmliyn4xBncGgalLjtVqeVu)

---



## Wireframe of the Application on different Devices



<img src="https://github.com/nerytompz/UdacityOpenSource/blob/Ateniola_Oluwatobi_Victor/Ateniola_Oluwatobi_Victor/assets/wireframe.png" width="100%">



## Model Description

I used a modified pre-trained resnet18 model architecture to create and train the currency classifier.



<img src="https://github.com/nerytompz/UdacityOpenSource/blob/Ateniola_Oluwatobi_Victor/Ateniola_Oluwatobi_Victor/assets/resnet18.png" width="100%">





---



## Tools Used

> The model was created and trained with PyTorch but then converted to ONNX (open ecosystem for interchangeable) and to PB (Protocol Buffers) for easier integration in a mobile application


- torch==1.1.0
- torchsummary==1.5.1
- torchtext==0.3.1
- torchvision==0.3.0
- onnx==1.5.0
- onnx-tf==1.3.0
- numpy==1.16.4
- matplotlib==3.0.3




```

pip install torch

pip install onnx

pip install onnx-tf

pip install pandas

pip install numpy 

pip install matplotlib

```



## References

1. [Discription of PyTorch models](https://pytorch.org/docs/stable/torchvision/models.html)

2. [How to convert a PyTorch model to onnx for easier mobile implementation](https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html)

3. [Explanation of the ResNet Architecture](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)

4. Secure and Private AI course on Udacity.
