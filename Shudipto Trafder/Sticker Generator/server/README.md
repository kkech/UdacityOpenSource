# Project: Sticker Generator
An Android App where user can generate a grid of sticker image, and user
can also save that image for further uses. In the backend gan model is
used to generate new Images and by using rest api those images bring to
android app.

## Future Possibilities
Create a sticker store, where user generate images and print from the shop.

# Schreenshorts
### GAN Model output
| First Image  | Second Image |
|---| ---|
|  ![First Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn1.png)  | ![Second Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn2.png) |

### Android App
| Splash Screen  | Main App Screen | Output Screen |
|---| ---| ---|
|  ![First Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn1.png)  | ![Second Image](https://github.com/Iamsdt/DeployBNDegit/blob/master/img/bn2.png) | |

# Architecture
This app contain 3 parts
1. Create Gan Model
2. Deploy in web (Rest API)
3. Develop Android App

## Gan Model
#### Framework and Datasets
**Framework**: Pytorch

**Datasets**:

**Datasets Provider**:

### Architecture
#### For Generator:
##### ConvTranspose2d Layer  
- Layers: 5
- Normalization: BatchNorm2d
- Activation Function: ReLU, TanH

#### For Discriminator:

#### Others:
- Epoch: 300
- Learning Rate: 1e-3
- Loss function: BCSLoss
- Optimizer: Adam
- Transformations:
    - Resize(64),
    - Center Crop(64)
    - RandomRotation(30),

#### Others Libraries
- Numpy
- PIL
- Matplotlib

#### Kernel Link
[BN digit with pytorch](https://www.kaggle.com/iamsdt/bn-digit-with-pytorch)

## Deploy in web (Rest API)
Gan model deployed in *heroku*

Web link -> 

**Language**: Python

A rest api is developed by using flask, the api takes number of images
as parameters and generate images, then make a grid image. And return
the image in base64 encoded string.

#### Used Libraries-
1. Flask
1. Pyorch
1. TorchVision
2. Numpy
3. Gunicorn
4. Imageio


## Develop Android App
For Developing Android App **Android Studio** is used.

**Language**: Kotlin

This app contain 4 screens -
- Splash Screen
- Main Screen
- Output Screen
- About Screen 

#### Design Pattern - MVVM
#### Used Libraries-
1. Live Data
2. ViewModel
4. Kotlin Coroutines
3. Retrofit
1. GSON
1. Timber
1. UCE-Handler

# Develpoer
**Shudipto Trafder**

Email: [Shudiptotrafder@gmail.com](mailto:shudiptotrafder@gmail.com)

Linkedin: [Shudipto Trafder](https://www.linkedin.com/in/iamsdt/)