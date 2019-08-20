#############
# Libraries #
#############

import numpy as np
from PIL import Image
import torch
from model import Generator
import PIL
from tqdm import tqdm
import os

MODEL_PATH = '../model/art-flower.pth'
IMAGE_INPUT_PATH = '../images/inputs/'
IMAGES_OUTPUT_PATH = '../results/'
sample_size = 16
z_size = 256
img_size = 256
gpu_available = torch.cuda.is_available()

def load_checkpoint():
    """
    Load a checkpoint file.
    
    returns
    -------
    A checkpoint file.
    """
    if gpu_available:
        GAN_checkpoint = torch.load(MODEL_PATH)
        print('Using GPU')
    else:
        GAN_checkpoint = torch.load(MODEL_PATH, map_location = 'cpu')
        print('Using CPU')
    return GAN_checkpoint

def load_generator(z_size, g_conv_dims, img_size):
    """
    Load the generator.
    
    returns
    -------
    The generator.
    """
    model = Generator(z_size, g_conv_dims, img_size)
    GAN_checkpoint = load_checkpoint()
    model.load_state_dict(GAN_checkpoint['G_model_state_dict'])
    
    return model

def create_sample(G, sample_size, z_size, seed):
    """
    Create a sample.
    
    returns
    -------
    A set saple z.
    """
    torch.manual_seed(seed)
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    if gpu_available:
        fixed_z = fixed_z.cuda()
    G.eval()
    samples_z = G(fixed_z)
    return samples_z

def convert_image(image):
    """
    Convert a tensor image into a numpy representation.
    
    returns
    -------
    A numpy image.
    """
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    image = ((image + 1)*255 / (2)).astype(np.uint8)
    return image

def save_image(image, name, size, path = '../results/'):
    """
    Save the images.
    """
    img = Image.fromarray(image)
    if size == 406:
        img = img.resize((256 + 150, 256 + 150), PIL.Image.ANTIALIAS)
    img.save(path + str(size) + '/' + name + '.png')


def generate_gif(G, size):
    """
    Generates a gif
    """
    gifs = []
    steps = 50 
    for k in tqdm(range(1), desc = 'Generating gifs'): 

        a = np.random.randint(z_size)
        b = np.random.randint(z_size)
        frames = []
        for l in range(steps * 2):
            xx = np.random.uniform(-1, 1, size = (z_size))
            theta = 3 % steps / (steps - 1)
            if l>=steps: theta = 1 - l % steps / (steps - 1)
            xx[a] = theta; xx[b] = 1 - theta
            xx.reshape((-1, z_size))
            fixed_z = torch.from_numpy(xx).float()
            if gpu_available:
                fixed_z = fixed_z.cuda()
            samples_z = G(fixed_z)
            img = samples_z.detach().cpu().numpy()
            img = np.squeeze(img, axis = 0)
            img = np.transpose(img, (1, 2, 0))
            img = ((img + 1)*255 / (2)).astype(np.uint8)
            img = Image.fromarray(img)
            if size == 406:
                img = img.resize((256 + 150, 256 + 150), PIL.Image.ANTIALIAS)
            frames.append(img)
        gifs.append(frames)
    return gifs

def save_gifs(G, size, generate = 'n'):
    """
    Save gifs.
    """
    if generate == 'y':
        print('Generating gifs')
        G.eval()
        gifs = generate_gif(G, size)
        for index, gif in tqdm(enumerate(gifs), desc = 'Saving gifs'):
            frames = gif
            frames[0].save('../results/' + str(size) + '/video.gif', format = 'GIF', append_images = frames[1:],
                           save_all = True, duration = 500, loop = 0)
