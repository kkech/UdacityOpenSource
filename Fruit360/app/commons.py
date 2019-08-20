import io

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def get_model():
  checkpoint_path = 'C:/Urvi/Private AI Scholarship/Kaggle/code/flask/newproj/app/model/Inception.pt'
  # model = models. densenet121(pretrained=True)
  model = models.inception_v3(pretrained=True)
  model.classifier = nn.Linear(2048, 196)
  model.load_state_dict(torch.load(
    checkpoint_path, map_location='cpu'), strict=False)
  model.eval()
  return model

def get_tensor(image_bytes):
  # my_transforms = transforms.Compose([transforms.Resize(256),
  #                       transforms.CenterCrop(224),
  #                       transforms.ToTensor(),
  #                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
  #                                             std=[0.229, 0.224, 0.225])])

    my_transforms = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)