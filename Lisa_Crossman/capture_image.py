#!/usr/bin/python3

from time import sleep
from datetime import datetime
from sh import gphoto2 as gp
import signal, os, subprocess
import time
import os.path
import smtplib
from email import encoders
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase


previous_state = False
current_state = False

def killgphoto2Process():
    p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    # search for the process which we want to kill
    for line in out.splitlines():
        if b'gvfsd-gphoto2' in line:
            #kill the process
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)

shot_date = datetime.now().strftime("%d-%m-%Y")
shot_time = datetime.now().strftime("%d-%m-%Y %H:%M%S")
picID = "PiShots"

def gphoto_settings():
    pg = subprocess.Popen(['gphoto2','--set-config','capturetarget=1'], stdout=subprocess.PIPE)
    print(pg.communicate())

clearCommand = ["--folder", "/store_00010001/DCIM/100NCD50/", "-R", "--delete-all-files"]
triggerCommand = ["--capture-image-and-download"]
#downloadCommand = ["--get-all-files"]
pwdCom = ["pwd"]

folder_name = shot_date+picID
print("folder name", folder_name)
save_location = "/home/pi/Desktop/gphoto/newimages"## + folder_name
print("save_location", save_location)

def createSaveFolder():
    try:
        os.makedirs(save_location)
    except:
        print("Failed to create new directory")
    os.chdir(save_location)

print("changed dir")
def captureImages():
    gp(triggerCommand)
    print("triggered")
    print("about to sleep")
    sleep(3)
    print("slept")
    print("slept, now downloading....")
#    downloadCommand
    print("download in place")
#    clearCommand
global j
j=0
def renameFiles(ID):
    global j
    for filename in os.listdir("."):
        j+=1
        print(filename, "filename")
        if len(filename) < 20:
            print("renaming files")
            if filename.endswith(".JPG"):
                os.rename(filename, (shot_date+picID+str(j)+".JPG"))
                print("renamed the JPG")
            elif filename.endswith("NEF"):
                os.rename(filename, (shot_date+picID+str(j)+".NEF"))
                print("renamed the NEF")

killgphoto2Process()
#gp(clearCommand)
gphoto_settings()
print("save folder bit")
i=0

while True:
   time.sleep(60)
   
   print("working dir")
   pw = subprocess.Popen(['pwd'], stdout=subprocess.PIPE)
   print(pw.communicate()[0])
   print("capturing")
   captureImages()
   ID = picID+"_{}".format(i)
   print("renaming to {}".format(ID))
   renameFiles(ID)
   sleep(10)
   i+=1
   photo_path = []
   server = smtplib.SMTP_SSL('smtp.gmail.com',465)
   shot_date = datetime.now().strftime("%d-%m-%Y")
   sender_address = "YOUR SENDER ADDRESS @gmail.com"
   sendphoto = []
   for filename in os.listdir("/home/pi/Desktop/gphoto/newimages/"):
        print(filename, "filename")
        if filename.endswith(".JPG"):
            sendphoto.append(filename)
            print("appended the JPG")
   print(sendphoto)
   photo_to_send = sendphoto
   for phot in photo_to_send:
       photo_path.append("/home/pi/Desktop/gphoto/newimages/{}".format(phot))
   print(photo_to_send, photo_path)
   server = smtplib.SMTP_SSL('smtp.gmail.com:465')
   fromaddr = "YOUR FROM ADDRESS"
   toaddr = sender_address
   server.login(fromaddr, "PASSWORD")

   # Create the container (outer) email message.
   msg = MIMEMultipart()
   msg['Subject'] = 'Subject photo'
   msg['From'] = fromaddr
   msg['To'] = toaddr
   body = "Its your photo!"
   print(photo_path)
   # Assume we know that the image files are all in PNG format
   for file in photo_path:
       # Open the files in binary mode.  Let the MIMEImage class automatically
       # guess the specific image type.
       print(file)
       fp = open(file, 'rb')
       img = MIMEImage(fp.read())
       fp.close()
       msg.attach(img)

   server.login(fromaddr, "PASSWORD")
   text = msg.as_string()
   server.sendmail(fromaddr, toaddr, text)
   server.quit()
   print("That's all sent for you!")

 # uncomment to carry out delete for these images   
#   pg = subprocess.Popen(['find','/home/pi/Desktop/gphoto/newimages/', '-type', 'f', '-name', '*.JPG', '-delete'], stdout=subprocess.PIPE)
#  print(pg.communicate())
