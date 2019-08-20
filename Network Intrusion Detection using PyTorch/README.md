# Network Intrusion Detection System using Pytorch

Networks are everywhere so we rely on the networks so much this makes it more prone to attacks as everything we do travels through some netowrk. It becomes vital to identify and protect the network from the attacks. There are many techniques available but with the capibilities of computers to learn anything themselves through machine learning algorithms can be applied to these system to create more robust systems which can detect any attack on the system. So here a system created which helps in indentifying attacks on the network.

## Technologies used:

We have used PyTorch, a python library for deep learning, to code for the project. PyTorch being simple to use comes with many robust features which helped creating the project so well.

## Algorithms:
The problem is a classification problem, using network statistics we have to identify whether a given condition is normal or intrusion. A Multi-Layered Perceptron algorithm is used which is a Deep Learning model, inspired from how human neurons work. The MLP classifier takes the input all the network statistics and at the output layer it has two outputs telling whether the activity is normal or attack.

## Dataset:
Deep learning works on the data. So we need to feed it with a unbiased dataset. Here dataset from KDD'99 cup for Network Anomaley Detection has been used.
It can be found on the folowing links:
1. Training set: https://drive.google.com/open?id=1yDeR2hLPySfeOghcEtUbZzbNtR4vxP0O
2. Test set: https://drive.google.com/open?id=1U89tsAPxkr1ELVAS5V_-0fVb7YVRl_HL

## Accuracy:
Model's accuracy lies around 85% which is a quite decent score as of now. It needs further exploration and research to tune the hyperparameters so that to provide the accuracy.

## Further work:
1. Increasing accuracy: For now accuracy score is quite decent but it needs imporvement to perform well.
2. Federated Learning: Pirvacy is an important concern for today's world and evevn in some critical applications like network, details can be used maliciously by some other users, so need to protect the privacy. For now applying federated learning is throwing some errors which is to be rrectified further and certainly we'll be able to correct those errors.

# Project workflow

**1. Data-preprocessing:** Computers don't understand the data in raw form we need to make it understand it by converting it into numbers that too in specific range. Data is cleaned from any noise and data is made unbiased, rdundancies are removed so we have a nice data to feed to our model. <br />
**2. Model:** An algorithm implementation where we feed the data and it gives out the result. MLP algorithm has been used here for the classification task. <br />
**3. Training:** Data is given to the network which in turn learn the patterns from the data and update its parameters so that it could give best results. <br />
**4. Testing:** A different data is given to the model as it getting trained so that to verify that the model prrdicts the results well or not. This, with test set we calculate the accuracy of the model.


# Project done by:
This project is indended for Udacity's **Secure and Private AI** Scholarship, for the project showcase challenge. 
Created by:
1. Nishant Bharat - @Nishant Bharat *(Slack Handle)*
2. Jaiki Yadav - @Jaiki Yadav *(Slack Handle)*
