# spaic_robotics
This repository is dedicated for showing the work of the "Cairo Local" subgroup within #sg_pytorch-robotics channel on Slack Work-space for Private and Secure AI challenge from Udacity.

# Project overview
We implemented the idea of "using federated learning to train rnn model with raspberry pi" that is discussed here: g.openmined.org/federated-learning-of-a-rnn-on-raspberry-pis/

In this project we try to approach the concept of video classification , we classify videos according to some labels , then use object recognition method , then train with rnn and then use federated learning , then use 2 raspberry pis and laptop to simulate as if we have two users and laptop as server tries not to grab the data but the models and then average the two models.
We use dataset from here: https://www.crcv.ucf.edu/data/UCF101.php

# Approaches followed 

1. Load the dataset
2. use opencv to build frames
3. make different datasets for each raspberrypi.
4.a. Use r-cnn as object recognition.
4.b. Use resnet pretrained model.
5. Use rnn model.
6. Use federated learning
7. Setup the laptop and the two raspberry pi to be in one network.

# Outcome of the project
 
 Videos are classified according to labels.
 Private models that are grabbed by the server and trained without the data.


# Project Participants
 
Following are the Slack handles of the members

| Name| Slack Name| Github
|--- | ---| --- |

|Nouran El-Kassas|@Nouran El Kassas|https://github.com/NouranElKassas
|Ahmed Magdy|@Ahmed MAGDY EISSA|
|Ahmed Thabit|@Ahmed Thabit|https://github.com/AhmedThabit
|Ziad Esam Ezat|@Ziad Esam Ezat|https://github.com/ZiadEzat
|Mahmoud Mahdi|@qursaan|https://github.com/qursaan
|Aya Khaled|@Aya Khaled|https://github.com/AyaKhaledYousef
|Mohammad Diab|@Mohammad Diab|https://github.com/depo-egy  


### Use case of our Lane Dection Project:

- <b> Can be used in mobile apps.</b>
   We will have secure and private models without the fear of data leak.
- <b> It can be used in digital media.</b>
   We can use this in digital media , media marketing and so on.
- <b> We can even use in movie making. </b>

### Using Pytorch Transfer Learning for car detection:
We tried using resnet model for detecting the frames that are taken from videos. But our main goals that we did not have the raspberry pi but used Dragonboard 410c instead because these were not available for uncontrollable reasons. Unfortunaletly we did not complete the hardware part , but we have installed ubunto on the kit.

### Our Future Enhancement(Our Ultimate Goal):
- Complete the rnn.
- Complete the hardware part.
- Make rest api.

