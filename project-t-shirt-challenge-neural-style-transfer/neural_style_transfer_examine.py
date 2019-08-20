# USAGE
#  python neural_style_transfer_examine.py --models models --image images/giraffe.jpg 

# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to directory containing neural style transfer models")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")
args = vars(ap.parse_args())

# grab the paths to all neural style transfer models in our 'models'
# directory, provided all models end with the '.t7' file extension
modelPaths = paths.list_files(args["models"], validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

# loop over the model paths
for modelPath in modelPaths:
	# load the neural style transfer model from disk
	print("[INFO] loading {}...".format(modelPath))
	net = cv2.dnn.readNetFromTorch(modelPath)

	# load the input image, resize it to have a width of 600 pixels,
	# then grab the image dimensions
	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image, set the input, and then
	# perform a forward pass of the network
	blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	# reshape the output tensor, add back in the mean subtraction,
	# and then swap the channel ordering
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)

	# show the images
	cv2.imshow("Input", image)
	cv2.imshow("Output", output)
	cv2.waitKey(0)