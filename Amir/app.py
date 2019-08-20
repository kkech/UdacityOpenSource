from flask import Flask,render_template, request
import phe as paillier
import numpy as np
from collections import Counter

app = Flask(__name__)

np.random.seed(12345)
print("Generating paillier keypair")
pubkey, prikey = paillier.generate_paillier_keypair(n_length=64)

print("Importing dataset from disk...")
with open('data/spam.txt','r') as f:
    raw = f.readlines()

spam = list()

for row in raw:
    spam.append(row[:-2].split(" "))
    
with open('data/ham.txt','r') as f:
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

print('Training Started')
model = HomomorphicLogisticRegression(spam[0:-1000],ham[0:-1000],iterations=10)
print('Training Done')
print('Prediction Started')

print('Encrypting model')
encrypted_model = model.encrypt(pubkey)
print('Encrypting model done')

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    print('Prediction Started')
    my_prediction = '';
    pred = 0
    final = {}
    if request.method == 'POST':
        comment = request.form['comment']
        data = comment[:-1].split(" ")
        #encrypting predictions
        encrypted_pred = encrypted_model.predict(data)
        try:
            #decrypting prediction
            pred = prikey.decrypt(encrypted_pred) / encrypted_model.scaling_factor
            if pred > 0:
                my_prediction = 'SPAM EMAIL'
            elif pred < 0:
                my_prediction = 'HAM (NOT A SPAM EMAIL)'       
        except:
            print("overflow")
    print("Prediction Done")    
    final = {'Prediction':my_prediction,'Score':pred}
    return render_template('result.html',prediction = str(final))

if __name__ == '__main__':
	app.run(debug=False)