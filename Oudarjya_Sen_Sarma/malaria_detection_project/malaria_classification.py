

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

# Step1: Load Dataset

dataframe = pd.read_csv("csv/dataset.csv")
print(dataframe.head())

# separate the features and labelhhhhhhjjmtts from the dataset
# Step2: Split into treaining and test data
# 


x = dataframe.drop(["Label"],axis=1) #get our features by dropping the Label column from our Data Set/ this operation is in place, so our original data set isnt in place!!!
y = dataframe["Label"] #then get the labels!!!

#next let submit our training and test datasets!!!
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

# train_test_split from sklearn!!!

# Step;4 build the model!!! as the dataset is relatively large we'll go for a Random Forest classifier!!! . If we had a smaller dataset then our first choice would have been SVm

model = RandomForestClassifier(n_estimators=100,max_depth=5)
# now lets fit our training Data
model.fit(x_train,y_train)

# and then we'll save our model using joblib because in real worls scenarios you'll be training your models initially and then will save them to make predictions in real time and not retrain your model everytime

joblib.dump(model,"rf_malaria_100_5")

# now lets make predictions on our test Dataset

# Step 5: Make predictions and get classification report!!!

predictions = model.predict(x_test)

print(metrics.classification_report(predictions,y_test))

# on running the above line we get the precision and recall
# the precision will give us how much error is in the model
# And 'Recall' will tell us how many times we're getting the error!!!s
# usualy a tradeoff between these two or something like inference score which is like a harbolic mean of these two give a better sense of how well your model  is performing!!!

# GO THROUGH OTHER PRE-PROCESSING TECHNIQUES AND FEATURE EXTRACTION TECHNIQUES AND TRY THEM OUT  IN THES EMODELS

# CHALLENGE: To build a neural network so that we can altogether skip out the feature extraction part we did!!! to get better results








