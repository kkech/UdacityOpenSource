from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.externals import joblib
import re
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Flatten, Conv1D, Dropout, Activation
# # GPU
# MAX_LEN = 100
# VOCAB_SIZE = 50000  # Size of vocabulary dictionary

# Hyperparams for CPU training
# CPU
# MAX_LEN = 50
# VOCAB_SIZE = 41000

app = Flask(__name__)
CORS(app)

data = [
    {
        "name": "Positive",
        "value": 0.75,
        "color": "#59c59f"
    },
    {
        "name": "Negative",
        "value": 0.25,
        "color": "#ea4335"
    }
]

# Training model using CNN and comparing accuracy
if tf.test.is_gpu_available():
    # GPU
    BATCH_SIZE = 128 # Number of examples used in each iteration
    EPOCHS = 3 # Number of passes through entire dataset
    VOCAB_SIZE = 50000 # Size of vocabulary dictionary
    MAX_LEN = 100 # Max length of review (in words)
    EMBEDDING_DIM = 40 # Dimension of word embedding vector

# Hyperparams for CPU training
else:
    # CPU
    BATCH_SIZE = 32
    EPOCHS = 3
    VOCAB_SIZE = 41000
    MAX_LEN = 50
    EMBEDDING_DIM = 40

# Model Parameters 
NUM_FILTERS = 300
KERNEL_SIZE = 4
HIDDEN_DIMS = 300

@app.route('/', methods=['POST', 'GET'])
def get_sentiment():
    # model_CN = joblib.load("CNN_sentiment.pkl")
    print(request.json)
    # Load trained Model
    # model_NaiveBayes = joblib.load("NB_twitter_sentiment")
    model = create_model()
    model.load_weights("./cnn_sentiment_weights.h5")
    # Sample String
    str = request.json['text']
    # p = model_NaiveBayes.predict([processTweet(str)])
    p = sentiment_analysis(str, model)
    sentiment = ""
    if p >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    print(p)
    return jsonify({'sentiment': float(p)})
    


def sentiment_analysis(raw_data, model_CN):
    data_tokenizer = pickle.load(open("tokenizer.pickle", "rb"))
   # Preprocessing the raw data
    tweet_np_array = data_tokenizer.texts_to_sequences([raw_data])
    tweet_np_array = sequence.pad_sequences(
        tweet_np_array, maxlen=MAX_LEN, padding="post", value=0)
   # Predict Sentiment
# Uncomment below to give sentiment only 1 or 0
#     sent = model.predict_classes(tweet_np_array)[0][0]
    # model_CN = pickle.load(open("tokenizer.pickle", "rb"))
    # print(tweet_np_array)
    # model_CN = joblib.load("CNN_sentiment.pkl")
    sent = model_CN.predict(tweet_np_array)[0][0]
    return sent


def processTweet(tweet):
    # remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # convert @username to AT_USER
    tweet = re.sub('@[^\s]+', '', tweet)
    # remove tickers
    tweet = re.sub(r'\$\w', '', tweet)
    # convert tweet to lowercase
    tweet = tweet.lower()
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # remove punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # remove words with two or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ')
    # remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet



def create_model():
    print("Building Model!")
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into EMBEDDING_DIM dimensions
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn NUM_FILTERS filters
    model.add(Conv1D(NUM_FILTERS,
                    KERNEL_SIZE,
                    padding='valid',
                    activation='relu',
                    strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(HIDDEN_DIMS))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model

if __name__ == '__main__':
    app.run(debug=True)
