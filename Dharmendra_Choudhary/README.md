# Projects done as Part of Secure and Private AI challenge 

## Tshirt Design

Using GAN and Neural Style Transfer to Create T-Shirt Designs

#### Current Results

|Model | Generated images |
|---| --- |
|[Conditional DCGAN](GAN_design/GAN_implementations/Conditional_DCGAN_MNIST.ipynb)|![C_DCGAN](GAN_design/images/Conditional_DCGAN/Fashion_MNIST_Tshirt.jpg) |
|CycleGAN | Work in Progress |
|[Neural Style Transfer](GAN_design/Neural_Style_Transfer/Neural_Style_Transfer.ipynb)| ![cat](GAN_design/images/Neural_Style_Transfer/cat_output_40000.jpg) |
|[Neural Style Transfer](GAN_design/Neural_Style_Transfer/Neural_Style_Transfer.ipynb)| ![tiger](GAN_design/images/Neural_Style_Transfer/tiger_ice.jpg) |
|[Fast Neural Style Transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style)| ![cat_candy](GAN_design/images/Fast_Neural_Style_Transfer/cat_candy.jpg) |
|[Fast Neural Style Transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style)| ![tajmahal_mosaic](GAN_design/images/Fast_Neural_Style_Transfer/tajmahal_mosaic.jpg) |


#### Notes
* Conditional DCGAN result is grid of 100 generated T-shirt images.
* CycleGAN implementation is completed, but need to train for few days for better results.
* For future work, Need to implement other variations of GANs and tuning for usecase of generating Designs.


## Secure and Private AI 

### Training Deep Learning Models with Differential Privacy
   
   Using MNIST data to train student model using n teachers models and performing PATE analysis
   
### Smart Keyboard via Federated Learning [Work in Progress]
   
   Training next word prediction model using federated learning with pysyft
