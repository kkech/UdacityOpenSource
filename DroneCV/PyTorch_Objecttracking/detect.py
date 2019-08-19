# USAGE
# Reference: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
# python3 detect.py

from __future__ import division

import time
import torch
from torch.autograd import Variable
from util import *
import argparse
from darknet import Darknet
import pickle as pkl
import random
from cv2 import VideoWriter, VideoWriter_fourcc


# Parse Command Line Arguments
def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="cfg/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="Video file to     run detection on", default="videos/drone2.mp4",
                        type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("cfg/coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# Check if cuda is available and use it
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

# Draw Rectangle
def write(x, results,  contours):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)

    return img

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (161, 155, 84)
greenUpper = (179, 255, 255)

# Detection phase
videofile = args.videofile  # or path to the video file.
vs = cv2.VideoCapture(videofile)
out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
start = time.time()
contours_rect = []
# allow the camera or video file to warm up
time.sleep(2.0)

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./detection.avi', fourcc, float(24), (1280, 720))

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1]
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break


	img = prep_image(frame, inp_dim)
	im_dim = frame.shape[1], frame.shape[0]
	im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

	if CUDA:
		im_dim = im_dim.cuda()
		img = img.cuda()

	with torch.no_grad():
		output = model(Variable(img, volatile=True), CUDA)
	output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

	if type(output) == int:
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		continue

	im_dim = im_dim.repeat(output.size(0), 1)
	scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

	output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
	output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

	output[:, 1:5] /= scaling_factor

	for i in range(output.shape[0]):
		output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
		output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

	classes = load_classes('cfg/coco.names')
	colors = pkl.load(open("pallete", "rb"))
	list(map(lambda x: write(x, frame, contours_rect), output))

	cv2.imshow("frame", frame)

	video.write(frame)
	key = cv2.waitKey(10)
	if key & 0xFF == ord('q'):
		break

vs.release()
video.release()

# close all windows
cv2.destroyAllWindows()