# Song recommendation for autonomous Vehicle

## Introduction
Everything from cars to the tap in your washroom is automated now-a-days. you do not need drivers in cars anymore. Your phone is booking appointments at the salon for you. Have you ever wondered how it would be if an AI assistance can recommend you song in Car on uttering some words and analysing your gender, age, mood and weather for better prediction.

Well, point to be noted here is, these uttered words are not singer name, song name, genre or word in lyrics. These can be anything from "I want to hear some fast hard band loud music which feels like storm" to "I am feeling bored play some awesome music". Now "awesome music" is different for a guy of an age 22 and man of age 36, Its different for a woman and a man. What if the weather is rainy or sunny, your desire to listen "awesome song" may vary. it will be different for a guy going through a breakup and a guy who just got a job in Facebook.

![1](https://user-images.githubusercontent.com/14244685/63367734-4fa56800-c39e-11e9-90c2-66ab72046fec.jpg)

### Why Users' query
Whatever the situation, mood, weather, age or gender. If users want to listen to some kind of song, that should be the first preference. AI has grown up fast, it can predict what your age, gender, etc is. But it still cannot read minds. Only user knows what kind of song one wants to listen. It does not matter if user seems 22 years old but if he wants to listen to old songs, then old song must be recommended to him.

### Why Users' age
Statistics obtained by survey shows that the song preference vary according to our age. And this is obvious too, your grandma may not love to listen, "I love it when you call me Senorita", or she may love( just kidding ). In simple words, our preference for song genre changes as our age passes.


![2](https://user-images.githubusercontent.com/14244685/63367735-503dfe80-c39e-11e9-9d21-8d433b72ca75.png)


### Why user gender
According to a survey Female love to listen Pop and Rock and Male love to listen Rap, Hip-Hop and Electronic music. So it's necessary to predict gender of user for better recommendations. (When male and female are together, you should ignore man's preference for betterment of both, just kidding )

![3](https://user-images.githubusercontent.com/14244685/63367738-503dfe80-c39e-11e9-837c-b79ae5cbc505.png)


### Why emotion

We prefer listening to different songs at different mood. If we are happy, we would go for some hip-hop kinda song, if sad, we would prefer listening slow song. So emotions directly affect the desire for song. 

Musical Element | Dignified/Solemn| Sad/Heavy | Dreamy/Sentimental|Serene/Gentle
------------ | -------------| ------------ | -------------|------------ 
| Mode | Major 4 | minor | minor 12| major 3 |
|Tempo | slow 14 | slow 12 |slow 16| slow 20 |
|Pitch| low 10 | low 19 | high 6| high 8 |
|Rhythm  | firm 18| firm 3 | flowing| flowing 2|
|Harmony | simple 3 | complex | - | simple 10|
|Melody | ascend 4 | - |- | ascend 3|


-------------------------

Musical Element | Graceful/Sparkling | Happy/Bright | Excited/Elated |Vigorous/Majestic
------------ | -------------| ------------ | -------------|------------ 
Mode | Major 21 | major | - | - 
Tempo | fast 6| fast 20 | fast 21| fast 6 
Pitch| high 16| high 6 | low 9| low 13
Rhythm  | flowing 8 | flowing | firm 2| firm 10  
Harmony | simple 12 | simple | complex 14| complex
Melody | descend 3 | - | descend 7| descend 8

<!---![4](https://user-images.githubusercontent.com/14244685/63367739-50d69500-c39e-11e9-80cc-bcec764ccbf3.png) --->


### Why weather

There are two things, one is that our mood depends on the weather, for examples, in summers people seem more happy while in winters people can be found sleepy, low in energy. Second, weather directly affect our preference too. let say in rain, we usually want to listen the song having words in lyrics related to rain. Direct relation between weather to genre could not be found but survey results says our desire to mode, tempo, pitch, rhythm, melody changes according to the weather.

### Huge factor : Privacy

Privacy has become a big factor nowadays. User would never want his/her music preference to be leaked. Recommendation model contain a lot of private information about user like query, image of the face, image of his location. There has been various adversarial attacking techniques like model inversion attacks which can be used to reverse engineer the model and input. Even image of your face can be obtained from model parameters and prediction probabilities.

![5](https://user-images.githubusercontent.com/14244685/63367740-50d69500-c39e-11e9-92dd-e129556f22d7.png)

Read this awesome paper for more details. 
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjf_NC-npHkAhXHfn0KHZvUAQMQFjAAegQIBhAC&url=https%3A%2F%2Fwww.cs.cmu.edu%2F~mfredrik%2Fpapers%2Ffjr2015ccs.pdf&usg=AOvVaw2Zl7-ovye12xyCeYm3tnhu

So here the novel idea was to combine Natural Language Processing, Computer Vision and privacy preserving techniques to provide better service with privacy than currently existing systems.

## Related Work

 There had been various work  done related to song recommendation using various techniques like naive bayes, simple logistic regression, multinomial naive bayes. But I have not seen much recent work using Neural network. There has also been some work on music recommendation on the basis of mood or feature detection from image. But combining various factors is a novel idea for better song recommendation.
 
The awesomeness of this project comes from privacy. I have not yet seen much privacy preserving techniques in autonomous cars. But it will play a huge role in the near future.

## Datasets

### NLP dataset

http://millionsongdataset.com/

Dataset has been derived from musiXmatch dataset, a subset of the famous Million Song Dataset (MSD) that includes the lyrics of 237,701 songs. 
The lyrics are stored in a bag-of-words format, where words are represented as positive integers. For efficiency, only the top 5000 most frequently uttered words in the total song vocabulary were included. Therefore, each song’s lyrics are delineated by the list of (token, n) tuples, where n is the number of instances of that token (i.e. word) in the song. We are also given a mapping of integer tokens to their corresponding word, which comes in lemmatized form.

The preprocessed dataset contains:

**"genre_to_mxm_track_dict.pkl"** is a dictionary with values as genre type and each value has key which contain song lyrics token. 
Corresponding to each key, there is a list which contain tag id of song. this dictionary does not contain whole lyrics of songs but just its tag id.

**"mxm_track_to_genre_dict.pkl"** file. This is a dictionary with keys as song tagids and each key contain only one value i.e. its genre.
it contains 396863 song tagids.

**“Querydata.npz”** contains sparse matrix of user query, in the form of 5000 dim vector.

### CV dataset

https://susanqq.github.io/UTKFace/

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc.

![6](https://user-images.githubusercontent.com/14244685/63367741-516f2b80-c39e-11e9-9ead-39c2dbe96b97.png)


## Technical Discussion

The project can be divided into three parts:
1. **Natural Language Processing** - NLP module will recommend top 10 songs for a query of user
2. **Computer Vision** - CV module will select top 5 songs out of songs recommended by NLP module on the basis of age, gender, emotion and weather.
3. **Encrypted Deep learning** -Encrypted DL will be used to encrypt model and data to preserve privacy.

### Natural Language Processing 

#### Feature selection

The goal was to extract semantic features from words in the lyrics of a song. 

First the lyrics is converted to vectors.

#### Why converted lyrics into vectors (word embedding)

In very simplistic terms, Word Embeddings are the texts converted into numbers and there may be different numerical representations of the same text. many Machine Learning algorithms and almost all Deep Learning Architectures are incapable of processing strings or plain text in their raw form.

#### Types of word embedding

1. Frequency based word embedding 
2. Probabilistic based word embedding

#### Count vector (current word embedding)

Consider a Corpus C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens will form our dictionary and the size of the Count Vector matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).
Then the word embedding is converted into Tfidf word embedding.

![7](https://user-images.githubusercontent.com/14244685/63367742-516f2b80-c39e-11e9-801c-29f13cc885f8.png)


#### Why TfidfVectorizer

This is another method which is based on the frequency method but it is different to the count vectorization in the sense that it takes into account not just the occurrence of a word in a single document but in the entire corpus. So, what is the rationale behind this? Let us try to understand.
Common words like ‘is’, ‘the’, ‘a’ etc. tend to appear quite frequently in comparison to the words which are important to a document. For example, a document A on Lionel Messi is going to contain more occurrences of the word “Messi” in comparison to other documents. But common words like “the” etc. are also going to be present in higher frequency in almost every document.
Ideally, what we would want is to down weight the common words occurring in almost all documents and give more importance to words that appear in a subset of documents.

![8](https://user-images.githubusercontent.com/14244685/63367743-5207c200-c39e-11e9-8a1c-1ca213db4395.png)
 

Now the dataset is ready to train model. 

#### Disadvantage of using above approach

Let's say the lyrics is “I love it when you call me senorita” the vector of lyrics would contain 1 (number of occurrences of that token) at all positions where tokens are {‘i’, ‘love’, ‘it’, ‘when’, ‘you’, ‘call’, ‘me’, ‘senorita’} in 5000 dim vector. 
This way only that word is recognized by model. But what if we somehow use features of word instead of frequency of word. For example, features of words “love”, “fondness”, “warmth”, “like”, “disposition” would have the same features. So if we see “love” somewhere and use its feature, we would be dealing with all similar kind of words to “love”. 
This approach is called Prediction based Embedding. Bert, Fasttext are some examples of Prediction based Embedding. 

I encourage to read this amazing document for more details.
https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

## Method 

This module is itself divided into two parts: 
Classification and Recommendation.

### Classification

Classification model is trained on song lyrics vector to predict genre of song. This solves two problems-
1. If a new song is added to database whose genre information is not given, it can be predicted from the model.
2. We could use only a part of the dataset because genre information was not given for some tracks. Those songs could be classified to genre using classifier model. 

## Recommendation

### Clustering

Now after converting to Tfidf word embedding. Word embedding would be of shape (number of songs, top words in vocab) i.e. (29512, 5000)
Each row is a song vector. These vectors are clustered in 20 clusters (you can use more number of clusters) using KMeans (unsupervised Clustering)
For understanding KMeans clustering go to this link.
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
If we would have clustered songs according to genre, we would have forced songs to be in particular genre. Since song lyrics sometimes are not pure “rock” or “electronic” we have created 20 clusters using unsupervised method i.e kmeans. Leaving on each song lyrics to find similar kind of vector and make cluster without giving genre information. 
After clusters are obtained, each cluster would have its mean vector( centroid ). 

### Recommender model

#### Training

“Querydata.npz” is the dataset containing user query for particular song. 
Recommender model would be trained to map query vectors to the cluster which contain respective song. 
Say, a is query for song b which is in cluster c, then input would be vector a and target would be mean vector of cluster c.

#### Prediction

When a query is passed in trained model, it would predict a cluster. Top 10 songs along with the genres would be recommended out of that cluster after calculating Cosine similarity between query vector and all the songs in that cluster. 

![9](https://user-images.githubusercontent.com/14244685/63367728-4e743b00-c39e-11e9-9c1d-3678d7a74c56.png)

Here NLP task is completed. 

## Computer Vision Module

This module contains four parts :
1. Face to age classification
2. Face to gender classification
3. Face to emotion classification
4. Image to weather classification

### Face to gender classification 

Age is predicted by model and genre top genre is selected. According to survey the preference of genre is:
Genre = [pop , rock, heavy metal, electronic, hip hop, other]
female = [47, 38, 8, 20, 20, 25] 
male = [8, 20, 14, 30, 30, 27]


![10](https://user-images.githubusercontent.com/14244685/63367729-4e743b00-c39e-11e9-8c9a-d8843a6b8a87.png)


### Face to age classification

Age is predicted by model and top genre is selected.

![11](https://user-images.githubusercontent.com/14244685/63367730-4f0cd180-c39e-11e9-9d2a-05d59228b719.png)


### Face to emotion classification

There is no statistics on internet of emotion versus genre. There could be found relation between harmony, pitch, rhythm, loudness but not genres. This feature would be added to the project as soon as some statistics could be found on the web. 

### Weather to genre classification

For weather, there could not be found any relation between weather and genre. This module is also not created because no dataset could be found for the same. 

### Combine

Using the age and gender prediction from cv models. Both the statistics are combined by simple multiplication and top three preferred genres are predicted.
Top 5 songs out of top 10 predicted by NLP model are selected on the basis of commom genre predicted by CV and NLP models.
These 5 songs are most suitable for recommendation. 

![12](https://user-images.githubusercontent.com/14244685/63367732-4f0cd180-c39e-11e9-98f2-c5818ed5ceaa.png)


## Encrypted Deep learning 

All the implemented Computer vision model are trained on encrypted data. Model is also encrypted for prediction. So that model owner is unaware of test data and data owner could not predict model parameters. 
Hence privacy is preserved. 
For more details relating encrypted deep learning read this amazing blog.
https://blog.openmined.org/encrypted-deep-learning-classification-with-pysyft/


![13](https://user-images.githubusercontent.com/14244685/63367733-4fa56800-c39e-11e9-8ffa-6fc397360f60.png)

I couldn't mention lyrics of songs but only track id as the dataset was already in token-frequency dictionary format. Song can be retrived from MusiXmatch dataset using the track id. 

## Problems faced

1. Model could not be trained on full dataset, since Colab provided limited RAM. Hence they are trained on subset of data and small batch size.
2. Prediction based word embeddings are huge in size ( minimum 2.3 GB ) so could not be loaded on colab because of limited RAM.
3. No dataset could be found for Weather to genre and Emotion to genre.

## Future Work

### Natural Language Processing module 

1. Prediction based word embedding can be used instead of frequency based word embedding for lot better results. I have discussed the reason above.
2. Various state of the art pretrained models like BERT can be used. So transfer learning could be applied for better extraction of words.

### Computer vision module

1. Emotion and weather based models could be implemented for better song prediction of genre and in turn better song recommendation. 

## Contributors 

 Slack Handle | Contributor
------------ | -------------
 @Ashish Shrivastava | Ashish Kumar Shrivastava 
  @Labiba | Labiba Kanij Rupty  
  @Mahedi | Md. Mahedi Hasan Riday  
 @Mayank Devnani  | Mayank Devnani  
  @Mohona | Mahfuza Humayra Mohona  
  
  
