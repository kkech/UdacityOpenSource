

Repository for Facial Emotion Recognition Project for Udacity Secure and Private AI Challenge Scholarship 
Team:#sg_speak_german



# EmoAR
## Facial expression recognition and Augmented Reality (AR)

A project by team #sg_speak_german:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mateusz Zatylny, @Mateusz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Berenice Terwey, @Berenice  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Akash Antony, @Akash Antony  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calincan Mircea Ioan, @Calincan Mircea Ioan  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Joanna Gon, @Joanna Gon  

This project was planned and created as a team effort for the Facebook partnered Secure and Private AI challenge by Udacity and utilized the knowledge acquired in this course.

**Short description of our project:**

EmoAR is a mobile AR application (mobile device with ARCore support is required) that aims to recognize human facial expression in real time and to overlay virtual content according to the recognized facial expression. For example: 
Depending on the predicted facial expression, EmoAR would display randomized famous quotes about the expression, in AR. (Quotes can motivate people to take positive changes in their life.)
The live AR camera stream of a mobile device (Android) is input to a segmentation tool (using tiny YOLO) that detects faces in the video frames in real time. 
The detected areas with a face are fed into a model that was trained on the public FER dataset (from a Kaggle competition 2013). 
The facial expression of the detected face is determined in real time by using our trained model. Depending on the model prediction (the output result), different virtual content overlays the face and adapts to the face's position. This virtual augmentation of a face is done with Augmented Reality (ARCore). 

Since ARcore is only supported by a small number of Android devices, we also deployed the model to a web app using Flask and Heroku, but without the AR feature. 

![project-diagram](https://user-images.githubusercontent.com/23194592/63302823-6d12fd00-c2de-11e9-9f0b-9a3cc274b243.jpg)


**Impact of EmoAR and its further implications**

1. Text overlays displaying the detected facial expression: EmoAR might help people with Asperger syndrome and autism in learning about the expression of a face.
2.  A classifier of facial expressions (trained model) will enhance robotic projects related to therapies that help autistic people to socialize. For example: [How Kaspar the robot is helping autistic students to socialise](https://www.abc.net.au/news/2018-06-05/the-creepy-looking-robot-teaching-kids-social-skills/9832530?pfmredir=sm)
    
3. It may also be used as a HR tool and support recruiters in evaluating the overall confidence level of an interviewee by measuring the change in emotions during the interviewee's responses.
    
4.  3d models or graphics, artistic content: In social media apps like Instagram, this could be used to suggest emojis, avatars, artistic visual content which suit the detected facial expression, so that the user can take a selfie with overlaid emojis, avatars and/ or artistic visuals.

![Android-drafts](https://user-images.githubusercontent.com/23194592/63307737-c5062f80-c2ef-11e9-8cc6-d878bf6bd1b4.jpg)




**Get the ARcore Android app EmoAR:**[Download apk](https://github.com/kshntn/EmoAR/tree/master/apk)


Our -.apk Android app file is too big for the GitHub repo. Thus, we had to split it in 2 file pieces. How to install the Android app:
1.  Check if your device supports ARCore. A list of ARCore supported devices: [https://developers.google.com/ar/discover/supported-devices](https://developers.google.com/ar/discover/supported-devices)
2.  Install winrar 
3.  Download both files 
4.  Both files have to be in same directory
5.  Then, open part 1 and unpack
6.  Allow installations of “unknown sources” in the Settings. 

**Demo video below:** [https://www.youtube.com/watch?v=Ezcn6U7Bz2U]


<a href="https://www.youtube.com/watch?v=Ezcn6U7Bz2U" target="_blank"><img src="https://user-images.githubusercontent.com/23194592/63371074-abaec300-c383-11e9-8cb3-c22bcfdd14e0.jpg" alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


**Go to the web app EmoAR:** [https://emoar.herokuapp.com/]

Click below to test our web app


<a href="https://emoar.herokuapp.com/" target="_blank"><img src="https://user-images.githubusercontent.com/23194592/63371155-d7ca4400-c383-11e9-9699-e4289ea01667.jpg" alt="IMAGE ALT TEXT HERE" width="750" height="400" border="10" /></a>

**Existing problem:**

OpenCV and AR frameworks like Vuforia, ARKit, ARcore do not work well together, because the input video stream of the AR camera has to be shared with the other frameworks and/ or SDKs.

In our project we need to determine whether and where faces are located in a video frame. With OpenCV techniques, haarcascade etc, this would have been an easy task, but it would prevent us from using the camera stream for the Augmented Reality overlay.

**Our workaround:**

Instead of using OpenCV’s techniques, we access the AR camera stream, we use YOLO to determine a person in a video frame, we crop this AR camera image, convert it to a Bitmap and feed a copy of it as input in our custom Neural Net to determine the facial expression of this face in real time. Most tasks are done asynchronously, the rendering of virtual AR overlays is done by accessing the OpenGL thread.

**Applied learnings:**


We have not only applied Deep Learning techniques learnt from the Challenge course, but we also did research and applied new learnings:

-   Trained CNNs with PyTorch, Tensorflow/ Keras specifically for mobile applications and web applications
    
-   Trained the CNNs by using a custom architecture and alternatively using transfer learning
    
-   Model conversion from PyTorch to ONNX to Tensorflow to Tensorflow Lite, and Keras to Tensorflow Lite for use in mobile applications, Unity3d and Android
    
-   Development of a REST API by using Flask.
    
-   Model deployment to a web app Heroku
    
-   Model deployment to Unity3d and Android
    
-   Combination of Deep Learning and Augmented Reality (AR) in an Android app. Deep Learning determines the kind of visual AR content.
    
-   Sharing of the AR camera feed with other APIs to be able to input the AR camera feed to the CNN, rendering on the OpenGL thread

**Data Description:**

We used the FER2013 dataset from Kaggle for training. [ [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview) ]
It was prepared by Pierre-Luc Carrier and Aaron Courville and consists of grayscale facial images of size 48x48 px. The faces are segregated and categorized into 7 classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
A total of 28,709 examples was used for training the models, which were further validated with 3,589 examples.

**About model training:**


We experimented with and trained several pre-trained models of different architectures:

-   ResNeXt50,
    
-   Inception Resnet v2,
    
-   MobileNet v2,
    
-   DenseNet121,
    
-   ResNet101,
    
-   a custom CNN
    

  

![Screenshot from 2019-08-20 01-21-20](https://user-images.githubusercontent.com/23194592/63305974-30003800-c2e9-11e9-86c1-43c6d5b9e62f.png)

  

We experimented with

-   data augmentation, i.e. rotation, horizontal flip
    
-   unfreezing some of the last layers
    
-   SGD and Adam optimizers,
    
-   different learning rates
    
-   schedulers – MultiStepLR, ReduceOnPlateau
    
-   weight initialization of linear layers
-  started cleaning the FER dataset
    
-   trained with PyTorch for the web app and Keras for Unity3d and Android

**About model conversion:**

Initially, we wanted to deploy to an Augmented Reality app (iOS and Android) via Unity3d using TensorflowSharp to do the inference. Tensorflow -.pb files are supported by Unity3d.
The conversion chain: PyTorch → ONNX → Tensorflow -.pb
We also tried the recently released Inference Engine by Unity3d with the Unity3d Barracuda backend from the ML Agents Toolkit.
Due to incompatibility issues concerning the Tensorflow versions as well as our models’ architectures with Barracuda, we dropped the app development in Unity3d, the issues led to crashes of Unity.
We switched to the development in Android (Java) with Tensorflow Lite and ARCore.
In order to keep the model conversion chain small, we decided for the conversion of a Keras model to Tensorflow Lite. The conversion reduced the model size from 108 to 36 MB.

**About the Android project:**


We used the following Android APIs and frameworks:

-   android.graphics
    
-   android.opengl
    
-   com.google.ar.core
    
-   org.tensorflow.lite

to name but one

To overlay and to place virtual 3d content with ARCore, ARPoint Clouds and ARPlanes are currently used. Learn about a better approach in the last section “Next steps. How to improve the project”

**About the web app project:**


We developed a REST API by using Flask. The image uploaded by the user is input as parameter. After model inference on the server, the prediction is returned as a base64-encoded string to the Heroku web app.

**Next steps. How to improve the project:**


1.   Higher accuracy of the model
    
2.   Cleaning the FER 2013 dataset, a small dataset (28.000 grayscale images, 48x48) taken from Kaggle
    
			i.   the dataset is unbalanced
    
			ii.  some images are ambiguous and have mixed emotions.
    
			iii. some images are wrong, i.e. only showing loading icons etc.
    
			iv.  gather RGB color images for a dataset
    

  

3.  Improve the dataset with more number of classes.
    
4.  Use of a dataset of facial landmarks and/ or 3d meshes (3d objects), because currently the app works best, if flat images of faces are used for inference.
    
5.  Advanced cropping of the bitmap in order to do inference only with the detected face(s).
    
6.  Overlay virtual objects on multiple faces simultaneously and in real time. We hope to be able to do this properly with learnings of the Computer Vision ND :wink: .
    
7.  Creating models to overlay as described above: emojis for social media apps, text overlays, artistic visual content which suit the detected facial expression



To overlay and to place virtual 3d content with ARCore, ARPoint Clouds and ARPlanes are currently used. A better approach in this regard is:
Use the 3d mesh of the face that is generated by ARCore and ARKit for face tracking to overlay the virtual 3d content as well as for model inference. This requires a dataset that has been trained on such 3d face meshes, is labeled (expression classes) and also gives information about landmarks. (Currently, we are using AR Point Clouds and AR Planes for tracking, and such a special dataset is not publicly available.)


**Application of Secure & Private AI in this project and future prospects:**

1.  User's data and training data will not leave the device and individual updates will not be stored in any servers/data centers.
2.  The data for training the models in this project are used taking into consideration any copyright and privacy  issues.

For future release and improvement of our model with new high quality private images, we plan to incorporate federated learning and encryption in order to secure the prediction of our classification models:

The user’s device will download the current shared model from a cloud,  will improve it by learning from and training on  local data on the device. Then the modifications will be summarized as an update. This model update will be sent back to the cloud where it is averaged with other user updates in order to improve the commonly shared model. All communication will have to be encrypted. 
This way, Federated Learning enables end devices to collaboratively learn a commonly shared prediction model while keeping all the training data on device.

Why we have not applied PySyft yet: 

Problems with using Differential Privacy and Federated Learning in our dataset
1.  Issues with using Encryption, Federated Learning, CNN and BatchNorm layers in pre-trained models.
2.  No GPU support for PySyft resulting in longer training times[ [https://github.com/OpenMined/PySyft/issues/1893](https://github.com/OpenMined/PySyft/issues/1893) ]


