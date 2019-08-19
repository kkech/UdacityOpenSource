# UdacitySPAIC-SaFASiLe
Secure and Federated learning with American Sign Language

coded by Richárd Ádám Vécsey Dr. (Richard)

Udacity Secure and Private AI Final Keystone Challange


## SUMMARY:

This program learns gestures from american sign language (ASL). The dataset comes from Kaggle. With this program you are
able to train a neural network to identify hand gestures from ASL. Use it as a module your code or a standalone program.
My goal was bulding a small and fast neural network. You are very welcome to rebuild it. See variables part to refine the
code.

This program contains 3 big parts:

- neural network
    
- federated learning
    
- secure learning

The core concepts meet with the requirements from SPAIC Keystone Challange: ND185, Lesson 9, Chapter 8


## REQUIREMENTS:

PySyft 0.1.16a1

PyTorch 1.0.1

TorchVision 0.2.2


## INSTRUCTIONS:

You have to save the trained model if you want to use it next time.

For saving you have to use the 'torch.save(model.state_dict(), PATH)' process.

For loading you have to use the 'torch.save(model.state_dict(), PATH)' process.

PATH is the path of your model.

    
## VARIABLES:

***************
batch size:    Batch size during training. It's better if the size is higher than the number of result categories.

epochs:        Count of training epoch Higher numbers aren't always better since overfitting. Watch out, this model could be
               overfitted over 35-45 epoch. If you want to prevent this, use higher dropout value, or make a bigger model. 
               Default value: 1
               
learning_rate: Learning rate of optimizer. I use ADAM. Default value: 0.0001

size:          One dimension from the size of input pictures (28x28). You have to change this if you use other source with
               different sizes. Be careful, you have to transform your data before load it. Default value: 28
               
num_workers:   Number of workres during training. Default value: 1

***************
hidden_1:      Number of hidden layers. Be careful, this model contains fully connected layers. Default value: 256

hidden_2:      Number of hidden layers. Be careful, this model contains fully connected layers. Default value: 256

output:        Number of result categories. If you want to predict more gestures, you have to change it a higher value.
               Default value: 25
               
dropout:       Value of dropout rate, where 0=0% (no dropout) and 1=100%. The value must be between 0 and 1. Higher value
               means higher dropping to prevent overfitting. However higher value causes longen the training process.
               Default value: 0.2

***************


## LINKS:

dataset: https://www.kaggle.com/datamunge/sign-language-mnist

PyTorch documentation: https://pytorch.org/docs/stable/index.html

PySyft: https://github.com/OpenMined/PySyft


## SPECIAL THANKS:

My best friend, Axel helps me a lot about this code. He supervised the coding process and made me motivated. He is a deep
learning coder and participant of another Udacity course names Deep Learning with PyyTorch Nanodegree.


## OTHER:

@author: Richárd Ádám Vécsey Dr. (Richard)

@email: richard@hyperrixel.com

@github: https://github.com/richardvecsey


@contributed partner: Axel Ország-Krisz Dr. (Axel)

@email: axel@hyperrixel.com

@github: https://github.com/okaxel
