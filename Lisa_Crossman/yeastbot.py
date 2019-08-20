import argparse
import numpy as np
import imutils.contours
import cv2
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from time import sleep

# Get our options
parser = argparse.ArgumentParser(description='Object height measurement')
parser.add_argument("-i", "--image", required=True, help="file to process")
parser.add_argument("-w", "--width", type=float, required=True,
                    help="width of the left-most object in the image")
args = vars(parser.parse_args())

# load image and check
image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
if image is None:
    print('Could not open or find the image:', args.image)
    exit(0)

rects = []

#apply HSV, blur and increase the brightness to minimise background noise 
image = cv2.medianBlur(image, 5)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,_,_ = cv2.split(hsv_image)

H,S,V = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
#increase the V (brightness)
V = V * 3

hsv_image = cv2.merge([H,S,V])

#following lines to convert to greyscale in case that is successful for another set of images
image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
#greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#function to determine the colourfulness of a selected part of the image (the greenest or bluest part!)

def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))

	# compute rg = R - G
	rg = np.absolute(R - G)

	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)

	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))

	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)



contour_list = []

cols = ['green','blue']

# define the list of boundaries, this determines a set of RGB describing "green" and "blue" note that colours are stored as BGR 
boundaries = [
	([0, 10, 15], [70, 250, 200]),
	([10, 0, 5], [250, 30, 60])]

#initialise variable j
j = 1

#loop through the colour masks in list boundaries
for i, (lower, upper) in enumerate(boundaries):
      lower = np.array(lower, dtype="uint8")
      upper = np.array(upper, dtype="uint8")

      resized_image = cv2.resize(image, (720, 1280))
      mask = cv2.inRange(resized_image, lower, upper)
      blob = cv2.bitwise_and(resized_image, resized_image, mask=mask)

#debugging print statement
#      print("this is for {}".format(image_colorfulness(np.hstack([resized_image, blob]))))
      cv2.imshow("images", np.hstack([resized_image, blob]))
      cv2.waitKey(0)
#find the contours in a particular colour mask
      _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#debugging print statement
#      print("length of contours = ".format(len(contours)))
#loop through all of the contours found
      for j, contour in enumerate(contours):
          (x, y, w, h) = cv2.boundingRect(contour)
#ignore any bounding boxes that are less than a height of 150
          if  h < 50 and w < 20:
              continue
          else:
              bbox = cv2.boundingRect(contour)
#zero out the mask area
              contour_mask = np.zeros_like(mask)
              cv2.drawContours(contour_mask, contours, j, 255, -1)
#identify the qualifying coloured regions one at a time
              print("found {} in region".format(cols[i]))
              region = blob.copy()[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
              region_mask = contour_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
              region_masked = cv2.bitwise_and(region, region, mask=region_mask)

#debugging file saves for all of the found sections, uncomment to check all these files
#              file_name_section = "colourblobs-{}-hue_{}-region_{}-section.png".format(i, cols[i], j)
#              cv2.imwrite(file_name_section, region_masked)
#              print(" * wrote {}".format(file_name_section))

              # Extract the pixels belonging to this contour
              result = cv2.bitwise_and(blob, blob, mask=contour_mask)

              output_image = np.hstack([resized_image,blob])
              cv2.rectangle(output_image,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]) , (255, 0, 0), 2)
#              fgbg = cv2.createBackgroundSubtractorMOG2()
#              fgmask = fgbg.apply(output_image)
#              cv2.imshow("images", output_image)
#              cv2.waitKey(0)

#check the colourfulness of the found region 
              print("this is for {} its {}".format(j, image_colorfulness(result)))
#only keep the most colourful regions (removes regions which are just background noise)
              if image_colorfulness(result) > 2.5:
              # And draw a bounding box
                  bottom_left, top_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
                  cv2.rectangle(result, bottom_left, top_right, (255, 255, 255), 2)
#save files of specific regions
                  file_name_bbox = "colourblobs-{}-hue_{}-region_{}-bbox.png".format(i, cols[i], j)
                  rects.append(bbox)
#debugging print statement
#                  print("bottom_left", bottom_left, "top_right", top_right)
                  cv2.imwrite(file_name_bbox, result)
                  print(" * wrote {}" .format(file_name_bbox))


#debugging print atatement
#print(rects, "rects")

mmPerPixel = args['width']/rects[1][3]
print(mmPerPixel, "rects height val", rects[1], rects[1][3])

print("so height of balloon is rects[0][3] * mmPerpix = {}".format(rects[0][3] * mmPerPixel))

#display image and press any key to exit
resized_image = cv2.resize(image, (720, 1280))
cv2.imshow('image', resized_image)
cv2.waitKey()
