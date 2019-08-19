#Copyright (c) 2019 Intel Corporation.

#Permission is hereby granted, free of charge, to any person obtaining
#a copy of this software and associated documentation files (the
#"Software"), to deal in the Software without restriction, including
#without limitation the rights to use, copy, modify, merge, publish,
#distribute, sublicense, and/or sell copies of the Software, and to
#permit persons to whom the Software is furnished to do so, subject to
#the following conditions:

#The above copyright notice and this permission notice shall be
#included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
#LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image
import cv2
import os

   
def display_images(directory, numOfImages = 5):
    file_list = glob.glob(directory + "/*/*")
    indicies = random.sample(range(len(file_list)), numOfImages * numOfImages)    
    fig, axes = plt.subplots(nrows=numOfImages,ncols=numOfImages, figsize=(15,15), sharex=True, sharey=True, frameon=False)
    for i,ax in enumerate(axes.flat):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #Pick a random picture from the file list
        imgplot = mpimg.imread(file_list[indicies[i]], 0)
        ax.imshow(imgplot)
        ax.text(10,20,file_list[indicies[i]].split("/")[-2], fontdict={"backgroundcolor": "black","color": "white" })
        ax.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    

def resize_image(file, size=299):
    img = Image.open(file)
    img = img.resize((size,size))
    img.save(file)

def check_image(file):
    if not file.endswith(".jpg"):
        #Not ending in .jpg
        print("Deleting (.mat): " + file)
        os.remove(os.path.join(os.getcwd(), file))
    else: 
        flags = cv2.IMREAD_COLOR
        im = cv2.imread(file, flags)
        
        if im is None:
            #Can't read in image
            print("Deleting (None): " + file)
            os.remove(os.path.join(os.getcwd(), file))
        elif len(im.shape) != 3:
            #Wrong amount of channels
            print("Deleting (len != 3): " + file)
            os.remove(os.path.join(os.getcwd(), file))
        elif im.shape[2] != 3:
            #Wrong amount of channels
            print("Deleting (shape[2] != 3): " + file)
            os.remove(os.path.join(os.getcwd(), file))
            
        if os.path.exists(os.path.join(os.getcwd(), file)):
            f = open(os.path.join(os.getcwd(), file), 'rb')
            check_chars = f.read()
            if check_chars[-2:] != b'\xff\xd9':
                #Wrong ending metadata for jpg standard
                print('Deleting (xd9): ' + file)
                os.remove(os.path.join(os.getcwd(), file))
            elif check_chars[:4] != b'\xff\xd8\xff\xe0':
                #Wrong Start Marker / JFIF Marker metadata for jpg standard
                print('Deleting (xd8/xe0): ' + file)
                os.remove(os.path.join(os.getcwd(), file))
            elif check_chars[6:10] != b'JFIF':
                #Wrong Identifier metadata for jpg standard
                print('Deleting (xd8/xe0): ' + file)
                os.remove(os.path.join(os.getcwd(), file))
            elif "beagle_116.jpg" in file or "chihuahua_121.jpg" in file:
                #Using EXIF Data to determine this
                print('Deleting (corrupt jpeg data): ', file)
                os.remove(os.path.join(os.getcwd(), file))  
