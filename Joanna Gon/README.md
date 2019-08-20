# Smile Detector

by Joanna (@Joanna Gon)

## Summary

In this project I explain how to develop a classificator using deep learning which can decide if a person is smiling or not (similar to the HappyHouse challenge).
For an explanation of the project see the presentation (Smile Detection.pdf). If you like to work with the code, run JoannaGonSmileDetector.ipynb.


##  Task

The task of this project is the 'happy house task', to build a image classificator, which can decide wether a person is smiling or not (since in the 'Happy House' only happy people are allowed to enter).

In case you don't know this task yet, see: 
https://www.kaggle.com/iarunava/happy-house-datasetor 
https://github.com/Kulbear/deep-learning-coursera/blob/master/Convolutional%20Neural%20Networks/Keras%20-%20Tutorial%20-%20Happy%20House%20v1.ipynb

The original challenge comes with a small dataset and most solutions use small and simple CNN solutions.

In this project, I use a larger dataset and build a solution using pretrained neural networks, to show how well larger datasets and pretrained networks can improve the accuracy.

## Dataset

I use the VGG Faces dataset which contains a large amount of pictures of faces of famous persons. See:
http://www.robots.ox.ac.uk/~vgg/data/vgg_face/

To keep training time sufficiently low, I have used only 700 training and 100 testing images. But it could be trained with even more images from  VGGFaces

In addition to the image data, the project includes code for data augmentation to create even more training data (However, the model performs pretty well even without using data augmentation).

## Code

The project loads the dataset and then trains 5 different CNNs for classification:
1. a comparatively simple CNN as baseline
2. pretrained, simplified VGG16 (smaller dense layer than original CNN)
3. pretrained Inception V3
4. pretrained Resnet 50
5. pretrained Inception Resnet v2

As you can see in the presentation, the project can possibly be extended to extract the face out of a picture before classification by using a pretrained RCNN such as mask rcnn inception (pretrained on COCO dataset).
However, when VGGFace pictures are used, this is not necessary since these pictures are already cropped pretty well.

## Results

In conclusion this project shows that accuracy of smile detection can be increased greatly by using a better and larger training dataset (such as VGGFace) and by using pretrained CNNs. In that, Inception v3 and Inception Resnet v2 achieve the highest accuracy.
