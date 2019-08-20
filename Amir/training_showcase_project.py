# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:45:18 2019

@author: Amir

@inspired by iamtrask blog https://iamtrask.github.io/2017/06/05/homomorphic-surveillance/
Homomorphic Encryption
Paillier Cryptography using a probabilistic, assymetric algorithm for public key cryptography

Data and opensource python library -->

http://www2.aueb.gr/users/ion/data/enron-spam/
https://iamtrask.github.io/data/ham.txt
https://iamtrask.github.io/data/spam.txt

https://pypi.org/project/phe/
A Python 3 library for Partially Homomorphic Encryption

before Homomorphic Encryption
1. people can see predictions and modify it as per their requiement to false others, predictions get Faked 
2. people can modify the trained model weights which is a steal, public model will be reverse engineered

after Homomorphic Encryption
1. people cannot modify as preditions is now encrypted
2. people cannot modify the trained weights as model is now encrypted 
"""

import phe as paillier
import numpy as np
from collections import Counter

np.random.seed(12345)
print("Generating paillier keypair")
pubkey, prikey = paillier.generate_paillier_keypair(n_length=64)

print("Importing dataset from disk...")
with open('spam.txt','r') as f:
    raw = f.readlines()

spam = list()

for row in raw:
    spam.append(row[:-2].split(" "))
    
with open('ham.txt','r') as f:
    raw = f.readlines()


ham = list()

for row in raw:
    ham.append(row[:-2].split(" "))
    
class HomomorphicLogisticRegression(object):
    
    def __init__(self, positives,negatives,iterations=10,alpha=0.1):
        
        self.encrypted=False
        self.maxweight=10
        
        # create vocabulary (real world use case would add a few million
        # other terms as well from a big internet scrape)
        cnts = Counter()
        for email in (positives+negatives):
            for word in email:
                cnts[word] += 1
        
        # convert to lookup table
        vocab = list(cnts.keys())
        self.word2index = {}
        for i,word in enumerate(vocab):
            self.word2index[word] = i
    
        # initialize decrypted weights
        self.weights = (np.random.rand(len(vocab)) - 0.5) * 0.1
        
        # train model on unencrypted information
        self.train(positives,negatives,iterations=iterations,alpha=alpha)
        

    
    def train(self,positives,negatives,iterations=10,alpha=0.1):
        for iter in range(iterations):
            error = 0
            n = 0
            for i in range(max(len(positives),len(negatives))):
                error += np.abs(self.learn(positives[i % len(positives)],1,alpha))
                error += np.abs(self.learn(negatives[i % len(negatives)],0,alpha))
                n += 2

            print("Iter:" + str(iter) + " Loss:" + str(error / float(n)))

    
    def softmax(self,x):
        return 1/(1+np.exp(-x))

    def encrypt(self,pubkey,scaling_factor=1000):
        if(not self.encrypted):
            self.pubkey = pubkey
            self.scaling_factor = float(scaling_factor)
            self.encrypted_weights = list()
            for weight in model.weights:
                self.encrypted_weights.append(self.pubkey.encrypt(
                int(min(weight,self.maxweight) * self.scaling_factor)))
            self.encrypted = True            
            self.weights = None
        return self

    def predict(self,email):
        if(self.encrypted):
            return self.encrypted_predict(email)
        else:
            return self.unencrypted_predict(email)
    
    def encrypted_predict(self,email):
        pred = self.pubkey.encrypt(0)
        for word in email:
            pred += self.encrypted_weights[self.word2index[word]]
        return pred
    
    def unencrypted_predict(self,email):
        pred = 0
        for word in email:
            pred += self.weights[self.word2index[word]]
        pred = self.softmax(pred)
        return pred

    def learn(self,email,target,alpha):
        pred = self.predict(email)
        delta = (pred - target)# * pred * (1 - pred)
        for word in email:
            self.weights[self.word2index[word]] -= delta * alpha
        return delta
    
model = HomomorphicLogisticRegression(spam[0:-1000],ham[0:-1000],iterations=10)

#encrypt model
encrypted_model = model.encrypt(pubkey)

#encrypted model weights
print(encrypted_model.encrypted_weights)

# generate encrypted predictions. Then decrypt them and evaluate.
test = raw[1001]
data = test[:-1].split(" ")        
encrypted_pred = encrypted_model.predict(data)
#encrypted prediction
print(encrypted_pred)
#decrypted prediction
pred = prikey.decrypt(encrypted_pred) / encrypted_model.scaling_factor

'''
pred < 0 is ham
pred > 0 is spam
'''
print(pred)
        
        
        
        