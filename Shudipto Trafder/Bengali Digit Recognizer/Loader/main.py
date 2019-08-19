def load(path):
    import torch.nn as nn

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

    model = Net()
    return load_checkpoint(model, path)


def predict(model, input_image_client):
    from torchvision import transforms
    from PIL import Image
    import io
    import torch

    preprocess = transforms.Compose([transforms.Resize(180),
                                     transforms.ToTensor()])

    for input_img in input_image_client:
        img_tensor = preprocess(Image.open(io.BytesIO(input_img)))[:1]  # Just use 3 channels
        img_tensor = img_tensor.view(1, 1, 180, -1)

    with torch.no_grad():
        out = model(img_tensor)
        proba = torch.exp(out)
        top_p, top_class = proba.topk(1, dim=1)
        value = top_class.item()

    return [value, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def load_checkpoint(model, path):
    import torch
    # Make sure to set parameters as not trainable
    for param in model.parameters():
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
        check = model.classifier
    except AttributeError:
        check = False

    if check is not False:
        model.classifier = classifier
    else:
        model.fc = classifier

    # Extract others
    model.cat_to_name = checkpoint['class_to_name']
    model.epochs = checkpoint['epochs']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model


