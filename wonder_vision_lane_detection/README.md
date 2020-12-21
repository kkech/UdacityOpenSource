# Wonder_Vision_Lane_Detection
This repository is dedicated for showing the work of the "Lane Detection" subgroup within #sg_wonder_vision channel on Slack Work-space for Private and Secure AI challenge from Udacity.

# Project overview
As the name suggests, this project focusses on <b>identification of lanes of the road</b> which we achieved through this project using various approaches and frameworks including but not limited to <b>OpenCV, PyTorch etc</b>. This model is to be used in drones , self driving cars, surveillance systems , so its is primarily related to 'Safety'.

# Approaches followed 

1. Reading Images
2. Color Filtering in HLS
3. Region of Interest
4. Canny Edge Detection
5. Hough Line Detection
6. Line Filtering & Averaging
7. Overlay detected lane
8. Applying to Video

# Outcome of the project

![2019-08-20 16_41_50-Slack _ Mohammad Diab, Shudipto Trafder, shivu, Astha Adhikari, Droid, Oudarjya ](https://user-images.githubusercontent.com/50787118/63344940-3cc76f00-c36f-11e9-8858-1f0c54604efc.png)

![2019-08-20 16_41_44-Slack _ Mohammad Diab, Shudipto Trafder, shivu, Astha Adhikari, Droid, Oudarjya ](https://user-images.githubusercontent.com/50787118/63344949-3e913280-c36f-11e9-9b65-c01255802c4a.png)

# Project Participants
 
Following are the Slack handles of the members

| Name| Slack Name| Github
|--- | ---| --- |
|Astha Adhikari|@Astha Adhikari|https://github.com/adhikariastha5
|Oudarjya Sen Sarma|@Oudarjya Sen Sarma|https://github.com/oudarjya718
|Mohammad Diab|@Mohammad Diab|https://github.com/depo-egy
|Shiva Shankar|@shivu|https://github.com/shiv-u
|Vigneshwari Ramakrishnan|@Vigneshwari|https://github.com/drvigneshwari


## This Google document is the for meetups follow up:
https://docs.google.com/document/d/1fGN4T_ZJNQP5KHoIm5kUwVwMsGtgJyX9qYuI5ZCVQWc/edit


### Here's the Workflow of our project!!!

![BlockDiagramLast](https://user-images.githubusercontent.com/19780364/63348471-0858b080-c379-11e9-844e-f9b14b3b852c.png)

### Use case of our Lane Dection Project:

- <b> It can be used in drones</b>
   We can use lane detection in automatic drones which are used nowadays like medical,traffic control,monitoring and so on.
- <b> It can be used in air surveillance system.</b>
   We can use this lane detection in to get a survey of places affected by disasters or in general normal surveillance.
- <b> We can even measure the traffic crowd in particlular lane. </b>

### Using Pytorch Transfer Learning for car detection:
We tried using vgg model for detecting the car using transfer learning. Though we could not completely integrate it with our project, we have a kept a small notebook for the things we have tried.

### Yolo object detection in video:
We also tried the object detection in a video using yolo and the weights are in the drive link above.

### Our Future Enhancement(Our Ultimate Goal):
- Detecting urban and rural roads.
- Use car detection in roads for a particular lane and count frequency (real time video)
- Use this detection with other projects in #sg_wonder_vision channel to create a mass useful project.

