# Tshirt Design
Using GAN and Neural Style Transfer to Create T-Shirt Designs

## Current Results

|Model | Generated images |
|---| --- |
|[Conditional DCGAN](GAN_implementations/Conditional_DCGAN_MNIST.ipynb)|![C_DCGAN](images/Conditional_DCGAN/Fashion_MNIST_Tshirt.jpg) |
|CycleGAN | Work in Progress |
|[Neural Style Transfer](Neural_Style_Transfer/Neural_Style_Transfer.ipynb)| ![cat](images/Neural_Style_Transfer/cat_output_40000.jpg) |
|[Neural Style Transfer](Neural_Style_Transfer/Neural_Style_Transfer.ipynb)| ![tiger](images/Neural_Style_Transfer/tiger_ice.jpg) |
|[Fast Neural Style Transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style)| ![cat_candy](images/Fast_Neural_Style_Transfer/cat_candy.jpg) |
|[Fast Neural Style Transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style)| ![tajmahal_mosaic](images/Fast_Neural_Style_Transfer/tajmahal_mosaic.jpg) |




### Notes
* Conditional DCGAN result is grid of 100 generated T-shirt images.
* CycleGAN implementation is completed, but need to train for few days for better results.
* For future work, Need to implement other variations of GANs and tuning for usecase of generating Designs.