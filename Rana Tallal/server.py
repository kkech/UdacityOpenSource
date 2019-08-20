# import libraries
from tensorflow.keras.datasets import cifar10
from  tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Conv2D, MaxPooling2D
from  tensorflow.keras.layers import Dense, Flatten
from  tensorflow.keras.optimizers import SGD
num_classes = 10
input_shape = (1, 32, 32, 3)
# Define model 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', batch_input_shape=(1,32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, name='logit'))
# We are assuming the model has already been trained and the weights # are saved in a .h5 file
# load the pretrained weights
pre_trained_weights = 'cifar10.h5'
model.load_weights(pre_trained_weights)

import syft as sy
hook = sy.KerasHook(tf.keras)

AUTO = True
worker_1 = sy.TFEWorker(host='localhost:5000', auto_managed=AUTO)
worker_2 = sy.TFEWorker(host='localhost:5001', auto_managed=AUTO)
worker_3 = sy.TFEWorker(host='localhost:5002', auto_managed=AUTO)

model.share(worker_1, worker_2, worker_3)
model.serve(num_requests=5) # limit the number of requests to 5 

