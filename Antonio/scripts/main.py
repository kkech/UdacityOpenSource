#############
# Libraries #
#############

from helpers import load_generator, create_sample, convert_image, save_image, save_gifs
import numpy as np
from tqdm import tqdm
import argparse
import torch
import matplotlib.pyplot as plt

MODEL_PATH = '../model/GAN_checkpoint.pth'
img_size = 256
g_conv_dims = 256
z_size = 256


parser = argparse.ArgumentParser('This program allow you to generate flower images.')
parser.add_argument('-i', '--images', type = str, default = 1, help = 'Number of images to generate.')
parser.add_argument('-s', '--size', type = int, default = 256, help = 'Output size. Allowed value are 256 and 406')
parser.add_argument('-g', '--gif', type = str, default = 'n', help = 'Generate a gif using the images.')
arg = parser.parse_args()


def run():
    if int(arg.images) >= 1:
        model = load_generator(z_size, g_conv_dims, img_size)
        for k in tqdm(range(int(arg.images)), desc = 'Generating images'):
            seed_torch = int(torch.randint(1, int(10e5), (1,)).numpy())
            seed_numpy = int(np.random.randint(1, 10e5, size = 1))
            np.random.seed(seed_numpy)
            torch.manual_seed(seed_torch)
            z_latent = create_sample(model, 1, z_size, seed_torch)
            image = convert_image(z_latent)
            image = np.squeeze(image, axis = 0)
            save_image(image, 'image_' + str(k), int(arg.size), path = '../results/')
        save_gifs(model, int(arg.size), generate = arg.gif)
    else:
        print('Unable to produce images using {}. Insert a number equal or greater than 1.'.format(arg.images))
    
if __name__ == '__main__':
    run()