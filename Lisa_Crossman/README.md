

OpenCV for Raspberry Pi 4

Project for workshop with children to measure blowing up a balloon with yeast activity.

Procedure followed:

1.  Collected a small-medium plastic bottle with no lid
2.  Collected yeast, warm water and a heaped teaspoon of sugar
3.  Blew up balloon and let down three times to make it more pliable
4.  Put warm water into the bottle, added sugar and 10g yeast
5.  Attached the balloon
6.  Attached a long blue lego block of a known width to carry out the measurements


Gathered together an old DSLR camera and a tripod
Used the script capture_image.py to capture a set of images with the camera (a Nikon D50)

Used the script yeastbot.py to analyse the images with OpenCV on the raspberry pi 4.



Installation of OpenCV on Raspberry Pi 4 for this project:

Note: OpenCV can be installed on Pi using sudo apt-get install, however, this will NOT install the newer version of OpenCV required for this project.  Also PyPi (pip) does not have the correct version of OpenCV.   The project requires OpenCV v. 4 at minimum.  Therefore OpenCV 4 will need to be compiled from source.  OpenCV 4 was officially released in November, 2018.

These instructions can be followed for installation of OpenCV4 on Pi4:

https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/


On running capture_image.py

python3 capture_image.py

  - this script can email the photograph images to you, using a gmail account on which the security settings have been modified to allow scripts to interact with the gmail account.
  - Note that folder names in the script will need to be altered to the users required folder names
  - Note that the user would need to write their sender and recipient email addresses and password into the script
  - Alternatively the user can comment out the email lines to allow just saving of the images to card.


On running yeastbot.py:

The script requires the parameters -i (image filename) -w (width of the object in mm to create the measurements)

example:
python3 yeastbot.py -i DSC_0030.JPG -w 37


The script creates views of the following:

- image on the left and masked green section on the right
- image on the left and masked blue section on the right
- full image

Finally the script prints out the measured height of the green balloon in mm.
