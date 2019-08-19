# Project: Sticker Generator
An Android App where the user can generate a grid of sticker image, and the user can also save that image for further uses. In the backend **GAN** model is used to generate new Images and by using rest API those images bring to android app.

## Future Possibilities
Create a sticker store, where user generate images and print from the shop.

# Schreenshorts
### GAN Model output
| First Image  | Second Image |
|---| ---|
|  ![First Image](https://github.com/Iamsdt/StickerGenerator/blob/master/img/output.png)  | ![Second Image](https://github.com/Iamsdt/StickerGenerator/blob/master/img/output2.png) |

### Android App
| Splash Screen  | Main App Screen | Output Screen |
|---| ---| ---|
|  ![Splash Screen](https://github.com/Iamsdt/StickerGenerator/blob/master/img/device-2019-08-18-175641.png)  | ![Main Page](https://github.com/Iamsdt/StickerGenerator/blob/master/img/device-2019-08-18-175654.png) | ![Output screen](https://github.com/Iamsdt/StickerGenerator/blob/master/img/device-2019-08-18-175717.png) |

## Test
Download And install [Android App](https://github.com/Iamsdt/StickerGenerator/blob/master/app/release/app-release.apk)

Notebook: [Anim Generator Gan PyTorch](https://www.kaggle.com/iamsdt/anim-generator-gan-pytorch)

See Server Code: [Server Code](https://github.com/Iamsdt/StickerGenerator/tree/master/server)

Browse Server [Sticker Generator](https://anime-generator.herokuapp.com/)


# Architecture
This app contain 3 parts
1. Create Gan Model
2. Deploy in web (Rest API)
3. Develop Android App

## Gan Model
#### Framework and Datasets
**Framework**: Pytorch

**Datasets**: [anime-faces](https://www.kaggle.com/soumikrakshit/anime-faces)

### Architecture
| Generator | Discriminator|
|---| ---|
|5 ConvTranspose2d Layers | 5 Conv2d layers|
| Normalization: BatchNorm2d | Normalization: BatchNorm2d|
| Activation Function: ReLU, TanH | Activation Function: LeakyReLU, Sigmoid|

#### Hyper Parameters:
| Hyper Parameters|
|---|
|Epoch: 500 |
| Batch Size: 128 |
| Learning Rate: 0.0002 |
| Loss function: BCELoss |
| Optimizer: Adam |
| Betas: (0.5, 0.999)|
| Transformations: Resize(64), Center Crop(64), RandomRotation(30) |
| Image Size: 64 |
| Generator input: 100 |

#### Others Libraries
- Numpy
- PIL
- Matplotlib

#### Kernel Link
[Anim Generator Gan PyTorch](https://www.kaggle.com/iamsdt/anim-generator-gan-pytorch)

## Deploy in web (Rest API)
Gan model deployed in *heroku*

**Language**: Python

A rest api is developed by using flask, the api takes number of images
as parameters and generate images, then make a grid image. And return
the image in base64 encoded string.

| Used Libraries|
| --- |
| Flask
| Pyorch
| TorchVision
| Numpy
| Gunicorn
| Imageio

##### Code Link
See Server Code: [Server Code](https://github.com/Iamsdt/StickerGenerator/tree/master/server)

Browse API: [API link](https://anime-generator.herokuapp.com/)

## Develop Android App
For Developing Android App **Android Studio** is used.

**Language**: Kotlin

This app contain 4 screens -
- Splash Screen
- Main Screen
- Output Screen
- About Screen 

#### Design Pattern - MVVM
| Used Libraries
| ---
| Live Data
| ViewModel
| Kotlin Coroutines
| Retrofit
| GSON
| Timber
| UCE-Handler

#### Code link
See source code: [Sticker Generator](https://github.com/Iamsdt/StickerGenerator/tree/master/app)

Download And install [Android App](https://github.com/Iamsdt/StickerGenerator/blob/master/app/release/app-release.apk)

# Develpoer
**Shudipto Trafder**

**Slack Name:** @Shudipto Trafder 

Email: [Shudiptotrafder@gmail.com](mailto:shudiptotrafder@gmail.com)

Linkedin: [Shudipto Trafder](https://www.linkedin.com/in/iamsdt/)