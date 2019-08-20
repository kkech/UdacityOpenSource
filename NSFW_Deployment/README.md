# Obscenity Detection in Images 

The objective of this project is to build an obscenity detector for images. Using this detector, one could classify images between two categories,  **'not safe for work' (NSFW) and 'safe for work' (SFW)**, i.e. whether images have obscene content or not respectively.

## Project Application

While there's a general notion of what's obscene and what is not, this notion varies from one person to another. A user may feel that an image doesn't have any obscene content and would feel safe to post it online. Posting an indecent image online can have severe consequences on the community. 
In such cases, blocking the user comes across as the general solution. 
A genuine user who had felt it was safe to post, can't do anything much after getting blocked. This block, in turn, forces the user to shift to other social platforms.

An obscenity detection model can effectively solve such problems and come handy in such situations. 

This model can be used on social media platforms such as Facebook to prevent users from uploading an indecent image by reminding the user about this before the image gets uploaded on the platform.

<img src="https://raw.githubusercontent.com/lalwaniabhi/NSFW_Deployment/master/assets/appwork.png?token=AI2XX7HYSKTUVRNU4ILRK225LQK2M" alt="Application Use Case">

###### Image 1: Use case 

Using this application, we aim to keep the traffic of such obscene content minimal, boosting the overall user experience.

## Project Details 

### Dataset 

To train our model we have used the NSFW dataset available at Kaggle provided by Vareza Noorliko. Dataset is available [here](https://www.kaggle.com/drakedtrex/my-nsfw-dataset).

### Detection Model 

The model used is a custom model with a [ResNet101 backbone](https://arxiv.org/abs/1512.03385). We have trained our model using [FastAI](https://www.fast.ai/) library.  The model training code is available [here](https://www.kaggle.com/lalwaniabhishek/nsfw-project?scriptVersionId=19160785).

**The model is deployed using [Render](https://render.com/) and is available [here](https://isitnsfw.onrender.com) for testing purposes.** 


<img src="https://raw.githubusercontent.com/lalwaniabhi/NSFW_Deployment/master/assets/code.gif" alt="Project Video GIF">

###### Image 2: Project Video (This includes one not safe for work image (NSFW) to showcase model working. Inconvenience caused is regretted)

## Future Plans 

* We plan to build a similar text-based obscenity detector model. This model would allow classifying sentences as 'safe for work' (SFW) or not. It'll be used in conjunction with our present model and would remind users before posting unsafe content, be it images or text. 

* We would extend our current model to allow it to classify videos as SFW. 

The overall aim is to cover all possible content types posted on social media platforms. 

### User Privacy

The images predicted to be NSFW by our model can be very personal sensitive images belonging to the user. To protect such sensitive information, we intend to process the obscenity detection model on images at a user's device and not at the server. Executing the model at a user's device would enable us to preserve user privacy. 

## Acknowledgements 

We acknowledge Facebook and Udacity for giving us this opportunity to participate in the Secure and Private AI course challenge. 

## Contirbutors

* Abhishek Lalwani (@Abhishek Lalwani) 
* Abhishek Tandon (@Abhishek Tandon) 
