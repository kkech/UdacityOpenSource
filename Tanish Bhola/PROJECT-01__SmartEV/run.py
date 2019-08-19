
#---------------------------------------Object Detection Part (To get the count of cars from an Image)---------------------------------------

from imageai.Detection import ObjectDetection
detector = ObjectDetection()
import os

detector.setModelTypeAsYOLOv3()
detector.setModelPath('./assets/yolo.h5')
detector.loadModel()

custom = detector.CustomObjects(car = True)
detections = detector.detectCustomObjectsFromImage(custom_objects = custom ,input_image="./assets/car.jpg", output_image_path="./assets/car_Detected.jpg", minimum_percentage_probability=30)

cars = 0
for eachObject in detections:
	cars += 1
#Se have the count of cars under a variable named cars

#-----------------------------------------------------Object Detection Part COMPLETE !!!-----------------------------------------------------


#--------------------------------------------------------------Get current Time--------------------------------------------------------------

from datetime import datetime
now = datetime.now()
# AM = 0
# PM = 1
time = now.strftime("%H")
time = int(time)

if time > 12:
    time = time - 12
    dayNight = 1
    
else:
    dayNight = 0

#--------------------------------------------------------Get current Time COMPLETE !!!-------------------------------------------------------


#---------------------------------------------------------------Prediction Time--------------------------------------------------------------

''' Now we have all the inputs needed for our trained Model to predict the Dynamic Price [cars, time, dayNight]

Lets load the model first.
'''

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

inputSize = 3
outputSize = 1

class LinearRegression(torch.nn.Module):
    def __init__(self,inputSize,outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize,outputSize)
        
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(inputSize,outputSize)

state_dict = torch.load('./assets/checkpoint.pth')
#print(state_dict.keys())
model.load_state_dict(state_dict)

# Okay so we're done with loading the model
# Let's secure it

import torch as t
import syft as sy
hook = sy.TorchHook(t)

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
secure_worker =sy.VirtualWorker(hook, id="secure_worker")

from torch import nn
from torch import optim
import torch.nn.functional as F

encryptedModel = model.fix_precision().share(alice, bob, crypto_provider = secure_worker)

test = t.tensor([cars, time, dayNight])
test = test.type(t.FloatTensor)

encryptedTest = test.fix_precision().share(alice, bob, crypto_provider = secure_worker)
encrypytedPrediction = encryptedModel(encryptedTest)
dynamicPrice = encrypytedPrediction.get().float_precision()
dynamicPrice = round(float(dynamicPrice.numpy()),3)
print('*'*20)
print('\n'*8)
print("DYNAMIC PRICE = ",dynamicPrice)

# I'm assuming the station has 5 charging ports and itatakes around 20 min to TopUp the car
waitingTime = (cars/5) * 20

print("Approximate Waiting Period =",waitingTime,"min")
print('\n'*8)
print('*'*20)

