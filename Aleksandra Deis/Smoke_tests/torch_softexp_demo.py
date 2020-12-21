'''
Script for demonstration of the soft exponential activation function.
'''
# import utilities
import sys
sys.path.insert(0, '../')

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# import APL function from Echo
from echoAI.Activation.Torch.soft_exponential import SoftExponential

# create class for basic fully-connected deep neural network
class Classifier(nn.Module):
    '''
    Basic fully-connected network to test BReLU.
    '''
    def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # initialize SReLU
        self.a1 = SoftExponential(256)
        self.a2 = SoftExponential(128)
        self.a3 = SoftExponential(64)

    def forward(self, x):
        # make sure the input tensor is flattened
        x = x.view(x.shape[0], -1)

        # apply Soft Exponential unit
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a3(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

def main():
    '''
    Script for soft exponential activation demonstration.
    '''
    # create a model to classify Fashion MNIST dataset
    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data for Fashion MNIST
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data for Fashion MNIST
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    print("Create model with {activation} function.\n".format(activation = 'SoftExponential'))

    # create model
    model = Classifier()
    print(model)

    # Train the model
    print("Training the model on Fashion MNIST dataset with {} activation function.\n".format('SoftExponential'))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")

if __name__ == '__main__':
    main()
