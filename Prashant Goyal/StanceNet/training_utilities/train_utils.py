# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:24:20 2019

@author: myidispg
"""

import torch
import os

import numpy as np

from models.full_model import OpenPoseModel
from models.paf_model_v2 import StanceNet

import utilities.helper as helper
import utilities.constants as constants

def criterion(prediction, label):
    sum = torch.sum((prediction - label)**2)
    return sum

def train_epoch(model, criterion_conf, criterion_paf, optimizer, scheduler,
                dataloader, losses, device, print_every=5000):
    """
    This method trains the model for a single epoch.
    Inputs:
        model: The model object to be trained. Instance of OpenPoseModel class.
        criterion_conf: The criterion for confidence maps
        criterion_paf: The criterion for Parts Affinity Fields
        optimizer: The optimizer instance used for updating the model weights.
        scheduler: This is the PyTorch LR Scheduler object to update the LR.
            Reducing LR is necessary after each epoch because the network then 
            takes large steps and the losses increses and decreses erratically.
        dataloader: The dataloader with the custom Dataset.
        losses: A list in which to append the losses. 
            They will be used for plotting purposes.
        device: A torch.device() object. To see if training on cuda(GPU) or CPU.
        batch_size: The batch size to be used for training.
        print_every: After how many batches to print the loss. Default=5000
    Outputs:
        model: The model trained for one epoch.
        optimizer: The optimizer object after one epoch training. It is saved
            because this contains buffers and parameters that are updated as 
            the model trains.
        losses: The list of losses for this epoch.
    """
    
    # A variable to accumulate losses until print_every is triggered.
    running_loss = 0
    
    print('Training for an epoch. This will take some hours depending on ' \
          'system specs...')
    
    # Use the DataGenerator to generate batches of data and perform training.
    for batch_num, (images, conf_maps, pafs, mask) in enumerate(dataloader):
        
        
        # Convert all to PyTorch Tensors and move to the training device
        images = images.permute(0, 3, 1, 2).float().to(device)
        conf_maps = conf_maps.float().to(device).permute(0, 3, 1, 2)
        pafs = pafs.float().to(device).permute(0, 3, 1, 2)
        mask = mask.float().to(device).view(mask.shape[0], 1, mask.shape[1], mask.shape[2])
        
        # Perform a forward pass through the model
        outputs = model(images)
        
        loss = criterion_paf(mask * outputs[0], mask * pafs)
        loss += criterion_conf(mask * outputs[1], mask * conf_maps)
        
        # Outputs is a dictionary with 5 keys for each stage. Keys are 1, 2, 3, 4 and 5.
        # Each key has another dictionary. Each sub-dictionary has 2 keys:
        # 'paf' or 'conf' for storing output of respective stage.
        
        # Define loss for both conf and paf individually. They will be equal to
        # sum of losses of all stages.
#        loss_conf_total = 0
#        loss_paf_total = 0
#        for i in range(3): # Three stages: 1, 2 and 3
#            conf_out = outputs[i]['conf']
#            paf_out = outputs[i]['paf']
#            loss_conf_total += criterion(mask * conf_out, mask * conf_maps)
#            loss_paf_total += criterion(mask *paf_out, mask * pafs)
        # Calculate the total loss for both the outputs: PAF and Conf map.
#        loss = loss_conf_total + loss_paf_total
        # Set grads to zero to prevent accumulation.
        optimizer.zero_grad()
        # Calculate the grads of all the parameters with respect to loss.
        loss.backward()
        # Take an optimization step.
        optimizer.step()
        # Take the scheduler step.
        scheduler.step()
        # Add to running_loss
        running_loss += loss.item()
        # Append losses to the optimizer dictionary.
        losses.append(loss.item())
        
        batch_num += 1
        
        # Print statistics
        if batch_num % print_every == 0:
            print(f'Batch number: {batch_num}\tAverage loss:' \
                  f'{running_loss/print_every}')
            running_loss = 0
    print('Finished epoch.')
    
    return model, optimizer, losses
        
        
def train(dataloader, device, num_epochs=5, val_every=False, print_every=5000,
          resume=False):
    """
    Train the model. The training can either begin a new or start from the 
    latest saved state. In case of resuming, the model state dictionary, 
    optimizer state, losses etc. will be read from the disk. The path to 
    the dave file can be found in "utilities/constants.py". AFter every epoch, 
    the model along with optimizer state and losses will be saved to disk.
    Inputs:
        dataloader: The dataloader that provided data from the custom dataset.
        device: A torch.device() object. To see if training on cuda(GPU) or CPU.
        num_epochs: The number of epochs for which to train. Default=5.
        val: Determine if the validation set is being used or not.
        print_every: While training, this is used to print loss metric after 
            batches in an epoch.
        resume: To see if training is starting or resuming. If resuming, info 
            is read from the path in "utilities/constants.py".
    Outputs:
        Returns 0 if successful. Else None.
    """      
    losses = [] # Initialize empty loss.
    # Losses is a list of lists. Each sub list contains losses for an epochs. 
    # Hence the number of sub-lists is equal to the number of epochs for which
    # the model has been trained.
    
#    model = OpenPoseModel(num_joints = constants.num_joints,
#                          num_limbs=constants.num_limbs).to(device)
    model = StanceNet(n_joints=constants.num_joints,
                      n_limbs = constants.num_limbs*2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_conf = torch.nn.MSELoss()
    criterion_paf = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma = 0.1)
    
    latest_epoch = 1
    
    if resume:
        print('Looking for a trained model.')
        # Get the latest epoch number from the saved models.
        contents = os.listdir(constants.model_path)
        if len(contents) == 0:
            print('No trained model exists. Please start with val=True')
            return None
        for name in contents:
            epoch = int(name.split('_')[1])
            latest_epoch = epoch if epoch > latest_epoch else latest_epoch
        # Construct model path using the latest epoch.
        model_path = os.path.join(
                constants.model_path, f'stancenet_{latest_epoch}_epochs.pth')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        try:
            losses = checkpoint['losses']
        except KeyError:
            losses = []
    
    # Begin training for the specified number of epochs.
    print(f'Epoch {latest_epoch}')
    for epoch in range(latest_epoch, latest_epoch + num_epochs):
        epoch_losses = []
        model, optimizer, epoch_losses = train_epoch(model, criterion_conf,
                                                     criterion_paf, optimizer,
                                                     scheduler, dataloader,
                                                     epoch_losses, device,
                                                     print_every=print_every)
        losses.append(epoch_losses)
        print('Saving model after this epoch now.')
        # Create a checkpoint and save the latest state.
        checkpoint = {'epoch': epoch,
                      'model_state': model.state_dict(),
                      'optimizer_state': optimizer.state_dict(),
                      'losses': epoch_losses
                      }
        model_path =  os.path.join(
                constants.model_path, f'stancenet_{epoch}_epochs.pth')
        torch.save(checkpoint, model_path)
    return 0
