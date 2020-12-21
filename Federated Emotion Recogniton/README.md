# Emotion Recognition in tweets
## Summary:
```
Submitted to: Project Showcase Challenge of Udacity and for the study group, #sg_caffeine_coders.
Usecase: It is used to recognize emotions from texts. 
Dataset: EmotionIntensity Dataset by WASSA-2017
Implementations: Federated learning along with GRU and LSTM are used.
```

## Objective 
Ever felt confused reading what others are writing about what they want to say??? Wouldn't it be an amazing thing if we already get the emotion of someone just by reading their texts? Then, this project of ours will come handy for you. Here, we used *Gated Recurrent Unit* to recognize people's emotions from what they write on social media. However, we all understand how much privacy is important to everyone and by keeping that in mind, we used *Federated Learning* to secure user privacy. 

*This project was created during the Udacity Facebook Secure & Private AI challenge scholarship and is covering the topic 'Emotion Recognition in text'
which is the process of identifying human emotion in text messages like tweets on [Twitter](https://twitter.com/home).*

<!---For the data we used a dataset given WASSA-2017 Shared Task on Emotion Intensity.--->

<!--- The project will be submitted to the [Project Showcase Challenge](https://sites.google.com/udacity.com/secureprivateai-challenge/community/project-showcase-challenge#h.p_E1Kba6yZtw4O)     
by Udacity and to the study group #sg_caffeine_coders project challenge. --->


## Dataset
We used 
[Emotion Intensity Dataset](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html) for our project where the data is collected from twitter.

An interactive visualization of the Tweet Emotion Intensity Dataset is shown below which is taken from [TweetEmotionIntensity-dataviz](http://saifmohammad.com/WebPages/TweetEmotionIntensity-dataviz.html)

![Alt_Text](https://github.com/rupaai/caffeine_coders_emotion_recognition/blob/master/Emotion_Intensity_Dashboard.png "TweetEmotionIntensity-dataviz")

The dataset provides a training set, a development set, and a test set and on each set it contains the information of a user's tweet, its emotion and the intensity. Four different emotions were marked for the emotion data which are *Happy, fear, joy and, sadness*.

For our project, we used the training dataset to train our model and testing dataset to check our accuracy. Besides, we did various preprocessing of the data which will be discussed in detail on *Data Cleaning* section.

#### Reference:
WASSA-2017 Shared Task on Emotion Intensity. Saif M. Mohammad and Felipe Bravo-Marquez. In Proceedings of the EMNLP 2017 Workshop on Computational Approaches to Subjectivity, Sentiment, and Social Media (WASSA), September 2017, Copenhagen, Denmark.BibTex

### Notebooks:

  1. **downloadTweets.ipynb**: This notebook helps us to download the tweets with specific keywords like, "#udacity", "#pytorch", "facebookai", "#facebook" using Twitter's Streaming API. We can feed in our model the real time tweets using this API.
  
  2. **emotion_ecognition_with_federated_learning.ipynb**: In this notebook, we have tried to apply course content(Federated Learning) to our model. But with failed attempts only, since RNNs and LSTM/GRU are not supported by Syft yet.
  
  3. **emotion_recognition_dataset1.ipynb**: Our main notebook in which we have used the dataset for training and testing of model. We are able to achieve an Accuracy of 91.85% using GRU. 

  4. **emotion_recognition_dataset2.ipynb**: Trained our model on another dataset with 6 emotion labels, with two new emotions 'love' and 'surprise', this dataset was very unbalanced so we are not taking it for our main model's training although it was able to give us 93% Accuracy.
  
  5. **Training_Dataset_Cleaning.ipynb** and **Testing_Dataset_Cleaning.ipynb**: In these notebook the given datasets of the Tweet Emotion Intensity Dataset with the labelled emotions anger, fear, sadness and joy will be loaded as csv files from the original source. Afterwards they will be combined into one csv and then changed into a dataframe, where the tweets can be accessed and cleaned. The outputs are the cleaned datasets for training and testing data.

## Data Cleaning
When we tweet or chat, we tend to be using various characters which are not part of English alphabets. However, in Natural Language Processing, it is really hard for the model to learn from data when it contains so many versatile characters and doesn't produce a good accuracy on that. So, to improve our accuracy we cleaned the data in the following process:
1. We removed the unnecessary characters like '@' or '#' which are pretty common in Twitter tweets.
2. We also removed links as those don't bear any meaning without the content.
3. Furthermore, we removed all the unreadable characters and converted the text into lowercase.

For further clarity, we shared our [Data Cleaning Code](https://github.com/rupaai/caffeine_coders_emotion_detection/blob/master/Training_Dataset_Cleaning.ipynb) here.

## Data Preprocessing

  1. Since computers understand only numbers, or to be more precise just binary, we cannot just feed in our algorithms directly into the text and expect them to give us the desired output.
  2. Just like in case of images, we represent the pixel values in a range of 0 to 255 to represent the colour intensity and use it to feed as raw data to feed our model, in case of text we need to convert them into numbers/tensors.
  3. To achieve this, first we are making a corporus/dictionary out of all the tweets we have received in our dataset.
  4. Then we are using word to index and index to word mappings to map words to index and index to words respectively.
  5. We are using tokenization to represent a sentence and then those tokens(words) are mapped to their respective indices.
  6. And in the final step we are using Word Embeddings as our final input to the model. 
  7. Each word in the corporus is represent using a word vector of specfic dimensions.
  8. Representing words in forms of tensors help us find the similarity and dissimilarity between two words, since the ones having similar meaning lie close to each other and the ones having different meanings lie very far away from each other.

## Model Architecture
  1. When we deal with text/tweets we know that there is some context related to it. Because one word can have different meaning, so it becomes very important for us to remember the context of our sentence.
  2. Recurrent Neural Networks helps us get over this problem by taking into account the previous context for a current input point.
  3. But when the length of our sentences tend to be longer, RNNs start suffering from a problem of Dimining Gradients which affects the performance in a very adverse way.
  4. To overcome this limitation, we are using GRU, a Gated Recurrent Unit which helps us in remembering only that context information which is important. 
  5. We are using a Single-layer GRU as our model for our Showcase Project.

## Federated Learning
  1. We were very excited about applying the course contents to our Showcase Project and no wonder Federated Learning was our favorite part of the course so after getting a very good accuracy of 93% on our model we started working towards it.
  2. We spent a good amount of time to execute it, but errors were not ready to leave our code, we tried searching on Google and StackOverflow, we even took help of our fellow scholars, but we were still not able to get over them.
  3. We then asked our question in Openmind's Slack Coomunity and got to know that Syft does not supports RNN and LSTM as of now and the teams are still working on it. [Link](https://openmined.slack.com/archives/C6EEFN3A8/p1566138140348300)
  4. The news stuck us very hard but after discussing this issue with Palak in one of the #ama_sessions we decided that we will mention about our hard work and here we are sharing it with you. We hope that soon we will be able to apply Federated Learning to RNNs also.

## Deployment

The deployment is done with FLASK. It loads a screen where it is possible to enter a message. However, we couldn't finish the whole deployment part due to time-restriction. Our future plan is to deploy the model completely after submitting a message on the screen, it will be analyzed. When finished analyzing, it will show the results screen, saying the percentage of each emotion the message is carrying. For now, the Machine Learning Model is not implemented in the deployment code, it will only show the screens like below:

![Alt_Text](https://github.com/rupaai/caffeine_coders_emotion_recognition/blob/master/UI_template.png "FLASK Screen")

Currently, the application will scrape Twitter data and analyze it.

## Proposed Use Cases
  1. After training our model for Emotion Recognition we wanted to use it for real time Emotion Detection.
  2. We had tried of scraping all the tweets using Twitter's Streaming API to download all the live tweets mentioning the keywords, "#udacity", "#pyTorch#, '#facebookai" , "#facebook".
  3. After fetching the real time tweets, we can apply our model to the tweets which can identify user's emotions from text about Udacity.
  4. For example, we can analyse student's emotions related to the Deep Learning Nanodegree over time and plot it.
  5. We can also use it for some other topics. For example: Recently Facebook decided to launch its own Cryptocurrency. We could fetch all the tweets with keywords, "#Libra","#fbCrypto", "#facebookCrypto" and then analyse in real time how people are reactingto this news using our emotion recognition model.
  6. Emotion Recognition model has one more application to its name where we can use it to classify the emotions of the feedbacks that are coming on Udacity's website. We canclassify the feedbacks into "Happy Customer", "Angry Customer", "Sad Customer". This way it can help Udacity in understanding user emotions in a better way since understanding Customer Emotion is a very important aspect for any firms/organisations.
  7. We also wanted to create a UI where a person can put some text and submit it and our model will tell about the emotion of the text.


## Contributors 

 Slack Handle | Contributor
------------ | -------------
 @Sabrina | Sabrina 
  @Sumanyu Rosha | Sumanyu Rosha
  @sourav kumar | sourav kumar
  @Labiba | Labiba Kanij Rupty
  
## References 
  1. [Raw Dataset Emotion Detection Tweets](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)
  
  2. [Facebook Color Set](https://encycolorpedia.de/3b5998)
  
  3. [Cleaning Data](https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90)
  
  4. [Udacity Facebook Secure and Private AI Scholarship Challenge Course](https://www.udacity.com/facebook-AI-scholarship)
  
  5. [Modified GRU - Federated Learning](https://github.com/andrelmfarias/Private-AI/blob/master/Federated_Learning/handcrafted_GRU.py)
  
  6. [Our query on Openmined slack about the federated learning issue](https://openmined.slack.com/archives/C6EEFN3A8/p1566138140348300)
  
  7. [Understanding RNN, LSTM & GRU](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  
  8. [Pre-processing for NLP](https://medium.com/@makcedward/nlp-pipeline-word-tokenization-part-1-4b2b547e6a3)
  
  9. [Analysing Twitter Data](https://towardsdatascience.com/visualization-of-information-from-raw-twitter-data-part-1-99181ad19c)
  
