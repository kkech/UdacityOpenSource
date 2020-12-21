import pandas as pd
import cv2, os
import numpy as np
import csv
import glob

label = "Parasitized"
dirList = glob.glob("cell_images/"+label+"/*.png")
file = open("csv/dataset.csv","a")
 # now we'll iterate through all the files in our Dataset
# we'll be running the script once for each class and since there are two classes we'll be running twice
# we'll read the image and apply a gaussian blurr
# with the gaussian blurr we're trying to smoothen the imAGE bigger the mask more will be the blurring



for img_path in dirList:
  im = cv2.imread(img_path)
  im = cv2.GaussianBlur(im,(5,5),2)
  
  im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  
  #   THE VARIablr contour will contain all the contours thats detected
  ret,thresh = cv2.threshold(im_gray,127,255,0)
  contours,_ = cv2.findContours(thresh,1,2)
  
  file.write(label)
  file.write(",")
  
  for i in range(5):
    try:
      area = cv2.contourArea(contours[i])
      file.write(str(area))
    except:
      file.write("0")
      
    file.write(",")
      
  file.write("\n")
   

# import cv2,os
# import numpy as np
# import csv
# import glob

# label = "Parasitized"
# dirList = glob.glob("cell_images/"+label+"/*.png")
# file = open("csv/dataset.csv","a")

# for img_path in dirList:
  
#   im = cv2.imread(img_path)
	
# 	im = cv2.GaussianBlur(im,(5,5),2)



# 	im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# 	ret,thresh = cv2.threshold(im_gray,127,255,0)
# 	contours,_ = cv2.findContours(thresh,1,2)
	
# 	for contour in contours:
# 		cv2.drawContours(im_gray, contours, -1, (0,255,0), 3)
	

# 	cv2.imshow("window",im_gray)

# 	break


# 	file.write(label)
# 	file.write(",")

# 	for i in range(5):
# 		try:
# 			area = cv2.contourArea(contours[i])
# 			file.write(str(area))
# 		except:
# 			file.write("0")

# 		file.write(",")

	
# 	file.write("\n")


# cv2.waitKey(19000)
