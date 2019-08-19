import base64
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request
import numpy as np

sys.path.append(os.path.abspath('model'))
sys.path.append(os.path.abspath('templates'))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # flaten tensor
        x = x.view(x.size(0), -1)
        return self.fc(x)


def init():
    _model = Net()
    path = os.path.join('model', 'model' + "." + 'pt')
    return load_checkpoint(_model, path)


def load_checkpoint(_model, path):
    # Make sure to set parameters as not trainable
    for param in _model.parameters():
        param.requires_grad = False

    # Load in checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location="cpu")

    # Extract classifier
    classifier = checkpoint['classifier']
    # set classifier
    try:
        check = _model.classifier
    except AttributeError:
        check = False

    if check is not False:
        _model.classifier = classifier
    else:
        _model.fc = classifier

    # Extract others
    _model.cat_to_name = checkpoint['class_to_name']
    _model.epochs = checkpoint['epochs']

    # Load in the state dict
    _model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(device)

    return _model


def transform(file):
    img = Image.open(file).convert('L')
    img = img.resize((180, 180), Image.ANTIALIAS)
    # print(type(img))
    img = np.array(img)
    img = np.broadcast_to(img, (1, 1, 180, 180))
    # print(type(img))
    img_tensor = torch.from_numpy(img)
    # print(img_tensor.shape)
    return img_tensor.float()


# init flask app
app = Flask(__name__, template_folder='templates')

global model

model = init()


def convert_image(x):
    x = x.replace('data:image/png;base64,'.encode(), ''.encode())
    with open("output.png", "wb") as fh:
        fh.write(base64.decodebytes(x))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imdata = request.get_data()
    convert_image(imdata)
    x = transform('output.png')
    with torch.no_grad():
        out = model(x)
        proba = torch.exp(out)
        top_p, top_class = proba.topk(1, dim=1)
        value = top_class.item()
        return str(value)


if __name__ == "__main__":
    app.run()
