#import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.preprocessing.image import load_img,img_to_array
import syft as sy
#create a client
client = sy.TFEWorker()
worker_1 = sy.TFEWorker(host='localhost:5000')
worker_2 = sy.TFEWorker(host='localhost:5001')
worker_3 = sy.TFEWorker(host='localhost:5002')
#connect to the secure model
client.connect_to_model(input_shape, output_shape, worker_1, worker_2, worker_3)

# prepare the image for prediction
def predict(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img
filenames=['horse.jpg','bird.jpg','car.jpg']
actual_labels = [7,2,1]
# Query the model for obtaining private predictions
for i,filename in enumerate(filenames):
    img = predict(filename)
    res = client.query_model(img)
    print(f"predicted class for {filename}:{np.argmax(res)} and actual class : {actual_labels[i]}")
