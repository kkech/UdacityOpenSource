# Project : Face Detection


  ## Study Group : #sg_wonder_vision
     This is the repository for group project of #sg_wonder_vision Face Detection team
     in Secure and Private AI Scholarship Challenge from Facebook | Udacity.
     In this project, one-shot learning with Siamese Network is implemented using PyTorch.
     

## Objectives 

The objective of the project is to calculate the similarity between two facial images. We forseee this application being used for recognizing known faces like in a visual attendnce system or as a face based login.


## Getting Started

This project uses Siamese neural network to do one shot learning. 

**Project Dataset**
- For this project, we use the AT&T Database of Faces, which can be downloaded from [here](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). Once you have downloaded and extracted the archieve, it will show s1-s40 folders and each having 10 different images per person.
You can use any dataset. Each class must be in its own folder.


### Dependencies

This require PyTorch v0.4 or newer, and torchvision. The easiest way to install PyTorch and torchvision locally is by following the instructions on the [PyTorch site](https://pytorch.org/get-started/locally/). You'll also need to install numpy and jupyter notebooks, the newest versions of these should work fine.Using the conda package manager is generally best for this,
```
conda install numpy jupyter notebook
```
If you haven't used conda before, please [read the documentation](https://conda.io/en/latest/) to learn how to create environments and install packages. 

*If you don't want to download these dependencies locally*, you can follow up alternative [Google Colab](https://colab.research.google.com/)

## Demonstration of face detection project 
[Project.ipynb]

<img src="https://github.com/JauraSeerat/Wonder_Vision_Face_Detection/blob/master/assets/code4.gif" alt="Project GIF">

## Applications of our model:

Face detection and face recognition is being employed by many companies in various applications. We are building this project with a focus toward the education domain. 

The application would work as a visual attendance/login system allowing it to -
* Carry out the classroom attendance by recogninzing students present in the class. 
* Allowing students to use their face for logging in to online course platforms, such as Udacity.



## Implementations:

### 1.Face Recognition
Achieved by using **Siamese Network based on PyTorch**
- This implementation quantifies the similarity between two faces.

###### What is Siamese Network?
Siamese Network is a special type of neural network and it is one of the popularly used one-shot learning algorithms. One-shot learning is a technique where model learn from one training example per class.
![siamese network](https://github.com/JauraSeerat/Wonder_Vision_Face_Detection/blob/master/Siamese%20network.jpg)

###### Why we use Siamese Network?

Unlike the general Deep Neural Networks, which require loads of data to effectively train a model, using siamese neural networks it is possible to train with very less training data. This 'one-shot' approach for learning on the data perfectly fits our application of face recognition where there are many faces to recognize, but working with limited resources, we have to keep the dataset small.

Siamese networks work by comparing two images being passed from the same CNN based branch and then calculating how close those two images are. This allows us to recognize faces by comparing two facial images. 
To learn more about Siamese Networks follow up this [tutorial](https://medium.com/swlh/advance-ai-face-recognition-using-siamese-networks-219ee1a85cd5)

### 2. Face Detection:
Achieved by using **Face Evolve Classifier** . This is *High-Performance Face Recognition Library based on PyTorch*
- This implementation is based on [MTCNN]( https://arxiv.org/pdf/1604.02878.pdf) network.  
- It provides bounding box and facial landmarks information and works well for detecting faces be it an individual facial image or a group image. 

### 3. Face Detection using Webcam:
Achieved by using **openCV and Python**
- Using webcam and face detection models, we can track user in real time. 
- A possible use case for this can be to calculate the time in which the user interacts with a course platform, watching course videos. 

## Future Plans

Our base application is ready and working and now we'll be working on this project to take it further and deploy it. 

### Gathering Metrics using Differential Privacy

This application would allow to track students as they access the online course platform, enabling to gather metrics such as student attentiveness. These metrics would then be passed back to the course platform administrators. These metrics can be used to redesign course material so that it is better suited to students, providing a more of a 'personalized ecucation'. 

While collecting metrics is beneficial to both parties, student privacy is a major concern here. Using differential privacy we can make sure that student privacy is protected. 



##  Contributors:
- Abhishek Tandon (@Abhishek Tandon )
- Alejandro Galindo (@Alejandro Galindo )
- Seeratpal K. Jaura  (@Seeratpal K. Jaura) 
- Sourav Das (@Sourav) 
- Agata Gruza (@Agata [OR, USA])
- Rupesh Purum (@Rupesh Purum )
- Joyce Obi (@Joyce Obi)

## References
- [Face recognition using Siamese Network](https://medium.com/swlh/advance-ai-face-recognition-using-siamese-networks-219ee1a85cd5)- Suggested by @Abhishek Tandon
- [OpenCV Face Recognition](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/) - Suggested by @Agata [OR, USA] 
- [Face detection using OpenCV and Python: A beginner's guide](https://www.superdatascience.com/blogs/opencv-face-detection) Suggested by @Seeratpal K. Jaura
- [Face recognition with OpenCV, Python, and deep learning](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) - Suggested by @Agata [OR, USA] 
- [Siamese Network](https://innovationincubator.com/siamese-neural-network-with-pytorch-code-example/) - Suggested by @Alejandro Galindo
- [Face Evolve](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) - Suggested by @Abhishek Tandon
- [Image Siamese Network](https://slideplayer.com/slide/10937846/39/images/32/Network+Architecture:+Siamese+Network.jpg)

## [Meetups Updates](https://docs.google.com/document/d/1bwPe_K4xh2Awk_72c1o9JmxKXtl661ko203j7e2_VpM/edit?usp=sharing)

