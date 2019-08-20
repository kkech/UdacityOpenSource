import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_data(num_clients, dataloader, cmap="autumn"):
    
    points = np.empty((1,3))

    for data in dataloader :   
        vec, lb = data

        lb = lb.detach().cpu().numpy()
        point = vec.detach().cpu().numpy()

        lb = lb.reshape((-1,1))
        point = np.concatenate((point,lb), axis=1)

        points = np.concatenate( (points,point), axis=0 )      

    points = points[1:]
    colors = plt.cm.get_cmap(cmap, num_clients)

    plt.figure(figsize=(10,10))
    plt.scatter(x=points[:,0], y=points[:,1], c=points[:,2], cmap=colors)