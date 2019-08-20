# Introduction

This project involves detecting of Sign Language Gestures from a video and translating them. Existing works are doing this on pre-recorded videos, whereas we are trying to detect gestures in real time from a mobile front camera. Also, most of the datasets available online are just the sign language alphabets, they are accurate and perform well, however, they can't be used practically for translation of words.

It is not practical to use the alphabets. We need to detect Gestures, and to do that, We planned for this project and aimed to complete it in 2 phases:
1. Building a Pipeline for translation
2. Making the algorithm Differentially Private and Enable Federated Learning

# 1. Pipeline   
The pipeline will be used to finalize the preprocessing to be used and model architecture for making the predictions.    

**a. Preprocessing**     
This is capturing the input and turning it into a way that our model understands. We used OpenCV for this part.
Since we are detecting the gestures in a video, the raw data that will be input is in video form.    

Every video is made of images. When more than one image form a sequence, we see motion, these images are called as frames. For each video[, we split into frames.](https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/) 

Every frame will have the person, his hands, maybe his legs and some other objects. However, for us, only hands are of importance. If we don't remove the useless information from the image, our model will be considerably slow. Therefore, we need to perform [hand segmentation.](https://medium.com/@soffritti.pierfrancesco/handy-hands-detection-with-opencv-ac6e9fb3cec1)
This sequence of images will be input into our model, which will predict the gesture.   

**b. Model Architecture**  
Our model will take input of a sequence of images. However, it doesn't understand the individual images. So our model architecture is made up of 2 neural networks.   

**CNN**  
We'll use a CNN to get features from a frame. This frame has the segmented hand from the image. The CNN is a pretrained network and will output a 2D tensor. Each frame will have a 2D tensor for it so our final output will be  a 3D tensor (sequence of frames). 

**RNN**  
[RNN with LSTM](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) are widely used for sequence of data. They are used for video classification and also for prediction of next word in a text input. The RNN will take the 3D tensor and output the final prediction of the gesture class.  

This pipeline needs to be validated while it is being coded, for this purpose we need a pre existing dataset which has gestures and recorded videos of each gesture.  

The next challenge was to find a valid dataset for our problem statement. We found few datasets but finalized on using the [LSA64 Argentinian Sign Language Dataset.](http://facundoq.github.io/unlp/lsa64/) 

The dataset had gestures using both hands, for initial training purposes we're only using single handed gestures. The total number of gestures are 40.

Why can't we use this dataset for our final model training? That's because the researchers who recorded this dataset used special gloves on the hands, we need to figure out the pre processing of hands for different skin colours, also our model will be much more accurate with data from many people as there may be slight variations in the gestures. 

# 2. Differential Privacy and Federated Learning

This is a future scope of this project and we haven't started implementation.  
After our pipeline is complete and validated, we need to train our model with input from multiple users. We need to make a model that is differentially private, i.e. the model doesn't adapt to any single training instance. Also, the raw videos for gestures will have the faces of people, we have to use [Federated Learning ](https://blog.openmined.org/upgrade-to-federated-learning-in-10-lines/), i.e. train the model on the mobile devices and upload the updated weights to the server. 

For this purpose, we need a mobile app which will have features to record video and then choose the class. After a user has submitted the videos with corresponding classes, instead of uploading the videos, the model will be trained on the mobile device and submitted back to a server. This way, our sign language model will be able to train on many more instances. 

# Current Progress
Most of us are beginners in Deep Learning, so we're currently still in Phase 1, and working on model architecture. We have done some pre processing for the Open CV part of the pipeline, but we still need to adapt the preprocessing for different skin colours. The model architecture is complete and we used pre trained Inception v3 for CNN, but the training is taking too long for a single video, We weren't able evaluate the model within the deadline, but We're still working on it.

# Team Members
Shubhendu Mishra  
Munira Omar  
Bhadresh Savani  
Aarthi Alagammai  
Anju M Alphonse  
Ruchika Ahuja  
Marwa Qabeel  

We would like to thank existing works of Harish Thuwal (https://github.com/hthuwal/sign-language-gesture-recognition), and HTSeng (https://github.com/HHTseng/video-classification) as it helped us when we were stuck.