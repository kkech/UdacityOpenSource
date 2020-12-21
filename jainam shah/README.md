Udacity Open Source

Project Showcase Challenge

Federated Learning
Decentralized Data - experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of GPU Processing

I have Used two models MLP and CNN model on MNIST and CIFAR10 Dataset from torch module only.

Run
The MLP and CNN models are produced by:

    python main_nn.py

The testing accuracy of MLP on MINST: 92.14% (10 epochs training) with the learning rate of 0.01. 

The testing accuracy of CNN on MINST: 98.37% (10 epochs training) with the learning rate of 0.01.

Federated learning with MLP and CNN is produced by:

    python main_fed.py

See the arguments in options.py.

python main_fed.py --dataset mnist --num_channels 1 --model cnn --epochs 50 --gpu 0

Model Accuracy Results

1) 10 epochs training with the learning rate of 0.01

                    MNIST   CIFAR10
                
        FedAVG-MLP	85.66%	72.08%
  
        FedAVG-CNN	95.00%	74.92%
  
2) 50 epochs training with the learning rate of 0.01

                    MNIST   CIFAR10
                
        FedAVG-MLP	84.42%	88.17%
  
        FedAVG-CNN	98.17%	89.92%

