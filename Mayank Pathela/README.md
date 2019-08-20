# Sentiment-Analyzer

## How to Run

### For Model Training

- [Install Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)
- Run [GenericSentimentAnalyzer.ipynb](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/GenericSentimentAnalyzer.ipynb) for Model Training step by step on jupyter.
- Run [SentimentAnalyzer.py](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/SentimentAnalyzer.py) to collect tweets on Samsung.(Make sure you have Twitter Developers Account and twitter api credentials).
- Run [SamsungTweetsAnalysis.ipynb](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/SamsungTweetsAnalysis.ipynb) for using Trained model for Tweets on jupyter.

### For running Web Application

#### Steps:-

- Navigate to [Web Application](https://github.com/starkblaze01/Sentiment-Analyzer/tree/master/Web%20Application) folder.
- Run `yarn install`
- Open two terminals and run `npm start` in Web application folder and run `python server.py` in kanjo-server folder to start front-end and back-end.

### Web Application using Naive Bayes Model

#### Positive:

![alt text](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/Web%20Application/public/images/Sentiment_eg.png)

#### Negative:

![alt text](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/Web%20Application/public/images/Screenshot_eg1.png)

### Web Application using CNN Model

#### Positive:

![alt text](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/Web%20Application/public/images/Screenshot_3.png)

#### Negative:

![alt text](https://github.com/starkblaze01/Sentiment-Analyzer/blob/master/Web%20Application/public/images/Screenshot_2.png)
