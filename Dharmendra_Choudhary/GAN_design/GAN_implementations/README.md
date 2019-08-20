# GAN Implementations

1. [Basic GAN](Basic_GAN.ipynb) :- Implementation of [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) paper. Generator and Discriminator doesn't use maxout layer used in the paper.
2. [DCGAN](DCGAN.ipynb) :- Implementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
3. [Conditional DCGAN](Conditional_DCGAN_MNIST.ipynb) :- Implement [conditional GAN](https://arxiv.org/abs/1411.1784) using DCGAN architecture.

## GAN Results

### MNIST

|GAN | Generated images |Generated images GIF| 
|---| --- |----|
|[Basic GAN](Basic_GAN.ipynb)|![Basic](../images/Basic_GAN/MNIST_epoch_50.jpg)|![Basic](../images/Basic_GAN/MNIST_animation.gif)|
|[DCGAN](DCGAN.ipynb)|![DCGAN](../images/DCGAN/MNIST_epoch_20.jpg)|![DCGAN](../images/DCGAN/MNIST_animation.gif)|
|[Conditional DCGAN](Conditional_DCGAN_MNIST.ipynb)|![C_DCGAN](../images/Conditional_DCGAN/MNIST_epoch_20.jpg) |![C_DCGAN](../images/Conditional_DCGAN/MNIST_animation.gif) |



### Fashion MNIST

|GAN | Generated images |Generated images GIF| 
|---| --- |----|
|[DCGAN](DCGAN.ipynb)|![DCGAN](../images/DCGAN/Fashion_MNIST_epoch_20.jpg)|![DCGAN](../images/DCGAN/FMNIST_animation.gif)|
|[Conditional DCGAN](Conditional_DCGAN_MNIST.ipynb)|![C_DCGAN](../images/Conditional_DCGAN/Fashion_MNIST_epoch_20.jpg) |![C_DCGAN](../images/Conditional_DCGAN/FMNIST_animation.gif) |
