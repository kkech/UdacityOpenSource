# Privacy based AI on Thermal Cameras

### Combining Privacy Based Deep Learning with IOT

### By Joy Banerjee

Since Thermal cameras have wide range applications, specially in military. It is very crucial that a good amount of privacy must be maintained in order to increase security. And so my project will be based on identifying objects in thermal images and using privacy based AI to increase the security.
 
So below is my network architecture diagram of my project : 

![Network Diagram](https://raw.githubusercontent.com/joybanerjee08/UdacityOpenSource/master/Joy%20Banerjee/Network%20Diagram.png)
 
The main server has an encrypted model and the workers Alice and Bob have encrypted data. The basic idea is that : main server uses the encrypted data from Alice and Bob, to train its encrypted model. Then use that encrypted model to predict the object in the thermal image coming from the Raspberry Pi.

## Raspberry Pi

The Raspberry Pi 3B+ uses a Lepton 2.5 Thermal Camera. The images that come from the Thermal Camera, are in 16 bits. That means the image pixels go from 0 to 65536. The Lepton 2.5 Thermal Camera has a resoltion of 80x60 pixels, which is okay for hobby purpose such as this project, but not good for industrial use. The refresh rate is around 9 Hz, which is okay for hobby purpose. The images are brought down from 16 bit to 8 bit (0-255) for processing. 

![Setup](https://raw.githubusercontent.com/joybanerjee08/UdacityOpenSource/master/Joy%20Banerjee/setup.jpg)
 
 ## Sending Data from Edge
 
Since the Raspberry Pi is a weak device to run a Neural Network on, it is not treated as a worker. Instead, it is treated as an Edge Device, that sends the images to cloudlet for inferencing. The images are encrypted using a simple scrambling encryption [which I found here](https://github.com/AtheMathmo/ImageEncryptor). The algorithm originally used PIL, so I had to modify it to use OpenCV image. The main premise of the algorithm is this : 

Original Image : 
![Me](https://raw.githubusercontent.com/joybanerjee08/UdacityOpenSource/master/Joy%20Banerjee/human.png)

Encrypted Image :
![Me](https://raw.githubusercontent.com/joybanerjee08/UdacityOpenSource/master/Joy%20Banerjee/encryptedHuman.png)

(I know the images are small, but that's directly from thermal camera's output :P)

As you saw above, the algorithm scrambles the pixels before sending them to the cloudlet. The cloudlet then unscrambles the picture and runs inference on it. 

This is done to protect the images from being seen by anyone else.

## Cloudlet :

The cloudlet has an encrypted model, which is trained from encrypted data belonging to Alice and Bob. The encrypted model is a small 3 layer neural network. The neural network has been kept small, so that the cloudlet can deliver real time images, keeping the actual FPS of the camera to a minimum.

But the tiny neural network has achieved 92% accuracy on the testing images, which is more than enough for my experiment !
After the inferencing process, the detection is sent back to the edge device for demonstration purpose of this project. 

The training images were a Coffee Cup, Noodles, Human (me), My Hand, Phone, Laptop and Raspberry Pi. 

**The resulting demo gif :** 

![Me](https://raw.githubusercontent.com/joybanerjee08/UdacityOpenSource/master/Joy%20Banerjee/thermalTest.gif)

## Running the Program : 

The program uses OpenCV, Pillow, Flask, pyTorch, pyLepton and pysyft. Make sure you have all of them installed before proceeding.

### Running the Notebook :

The program can be run both on google colab, as well as local. The just keep executing the program cell by cell until you start the server. To use the pretrained Network, keep the file 'littleThermalNet.pt' in the same folder as the Jupyter Notebook. The training images are provided in 'thermalTrain.zip'. So just unzip it in the same folder as the Jupyter Notebook and you're good to go !

### Running the piThermal.py:

After you start the server, then start piThermal.py. The program will ask if you want to train the server with new images. For inferencing by default, answer 'n' to the above question, then an opencv frame will openup soon and you should see your image.  And if you want to train new images, then answer 'y' and if will collect all the images, the instructions will be given on screen. Press 'Space Bar' to take pictures, 'C' to change label of the input pictures and 'N' to quit taking pictures. After that, the program will send images to the server. Then the program will ask you if you want to train from scratch, to which if you answer 'y', then it will train the model from scratch, and if you answer 'n', it will train on the previous model (Transfer Learning). 

## Future Prospects : 

1. More powerful edge computing device, so that the edge can be treated as a worker as well.
2. Better thermal camera. 80x60 isn't enough to utilize the full fledged power of the thermal spectrum !
3. Better Neural Network : Using SSD can help us not only detect objects, but also localize them in images.

