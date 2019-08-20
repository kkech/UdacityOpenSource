# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:26:46 2019

@author: myidispg
"""
import argparse

import torch
import numpy as np
import cv2

import os

from utilities.detect_poses import get_connected_joints, find_joint_peaks
from utilities.constants import threshold

from models.paf_model_v2 import StanceNet

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str, help='The path to the video file.')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

video_path = args.video_path

# If the path has backslashes like in windows, replace with forward slashes
video_path = video_path.replace('\\', '/')
video_path = os.path.normpath(video_path)

video_path = os.path.join(os.getcwd(), video_path)

if os.path.exists(video_path):
    pass
else:
    print('No such path or file exists. Please check.')
    exit()
    
print('Loading the pre-trained model')
model = StanceNet(18, 38).eval()
model.load_state_dict(torch.load('trained_models/trained_model.pth'))
model = model.to(device)
print(f'Loading the model complete.')

# now, break the path into components
path_components = video_path.split('/')
video_name = path_components[-1].split('.')[0]
extension = path_components[-1].split('.')[1]

try:
    os.mkdir('processed_videos')
except FileExistsError:
    pass

output_path = os.path.join(os.getcwd(), 'processed_videos', f'{video_name}_keypoints_1.{extension}')
print(f'The processed video file will be saved in: {output_path}')

# Now that we have the video name, start the detection process.
vid = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_path, fourcc, 20.0,
                      (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'Video file loaded and working on detecting joints.')
frames_processed = 0
while(vid.isOpened()):
    print(f'Processed {frames_processed} frames out of {total_frames}\n', end='\r')
    ret, orig_img = vid.read()
    if ret == False:
        break
    orig_img_shape = orig_img.shape
    img = orig_img.copy()/255
    img = cv2.resize(img, (400, 400))
    # Convert the frame to a torch tensor
    img = torch.from_numpy(img).view(1, img.shape[0], img.shape[1], img.shape[2]).permute(0, 3, 1, 2)
    img = img.to(device).float()
    # Get the model's output
    paf, conf = model(img)
    # Convert back to numpy
    paf = paf.cpu().detach().numpy()
    conf = conf.cpu().detach().numpy()
    # Remove the extra dimension of batch size
    conf = np.squeeze(conf.transpose(2, 3, 1, 0))
    paf = np.squeeze(paf.transpose(2, 3, 1, 0))

    # Get the joints
    joints_list = find_joint_peaks(conf, orig_img_shape, threshold)
    # Draw joints on the orig_img
    for joint_type in joints_list:
        for tuple_ in joint_type:
            x_index = tuple_[0]
            y_index = tuple_[1]
            cv2.circle(orig_img, (x_index, y_index), 3, (255, 0, 0))
            
    # Upsample the paf
    paf_upsampled = cv2.resize(paf, (orig_img_shape[1], orig_img_shape[0]))
    # Get the connected limbs
    connected_limbs = get_connected_joints(paf_upsampled, joints_list)
    
    # Draw the limbs too.
    for limb_type in connected_limbs:   
        for limb in limb_type:
            src, dest = limb[3], limb[4]
            cv2.line(orig_img, src, dest, (0, 255, 0), 2)
            
    out.write(orig_img)
    frames_processed += 1

vid.release()
print(f'The video has been processed and saved at: {output_path}')