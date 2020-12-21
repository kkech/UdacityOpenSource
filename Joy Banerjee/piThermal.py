
import numpy as np
import cv2
from pylepton import Lepton
import io
import base64
import requests

url = 'http://192.168.1.5:5000/'

def basic_encrypt_scramble(im):
    y_size,x_size = im.shape
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    for x in range(x_size):
        for y in range(y_size):
            pixel = im[y, x]
            coprime_pixel = ((c+1 if c % 2 == 0 else c) for c in pixel)
            im[y, x] = tuple(int(c**43 % 256) for c in coprime_pixel)
    im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    return im

if raw_input("Want to take new pictures and train the server ? (y/n) ") == "y":
    while True:
        name = ""
        ct = 0
        with Lepton() as l:
            cap,_ = l.capture()
            if name == "":
                name = raw_input("Who is in this picture ?")
                print("\n\nTo stop taking pictures, type 'n'\nTo change person, type 'c'\nElse just press space\n")
            else:
                if not os.path.exists('./dataset/'+name):
                    os.makedirs('./dataset/'+name)
                cv2.normalize(cap, cap, 0, 65535, cv2.NORM_MINMAX)
                np.right_shift(cap, 8, cap)
                cap = np.uint8(cap)
                height, width = cap.shape[:2]
                
                cap = cv2.flip( cap, -1 )
                img = cv2.resize(cap, (8*width, 8*height), interpolation = cv2.INTER_CUBIC)
                cv2.imshow('image', img)
                decision = cv2.waitKey(1) & 0xff
                if decision == 32:
                    cv2.imwrite('./dataset/'+name+'/'+str(ct) + ".png", cap)
                    ct+=1
                if decision==ord('n'):
                    break
                if decision==ord('c'):
                    name=""
                else:
                    break
    cv2.destroyAllWindows()
    ct = 1
    dicp = {}
    directory = list(os.walk('dataset'))
    for name in directory[0][1]:
        dicp[name] = []
        for files in directory[ct][2]:
            dicp[name].append('./dataset/'+name+'/'+files)
        ct+=1

    gray = io.BytesIO()
    for name in dicp:
        for file in dicp[name]:
            try:
                gray.write(cv2.imencode('.png', basic_encrypt_scramble(cv2.imread(file,0)))[1].tostring())
                post_fields = dict(image = base64.b64encode(gray.getvalue()),name=name)
                gray.seek(0)
                gray.truncate(0)
                cloudMessage = requests.post(url+'savepic',data=post_fields, allow_redirects=True)

                if cloudMessage.content == 'NOK':
                    print('This file had an error :',file)
            except Exception as e:
                print('There was an error : ',e)
    
    cloudMessage='' 
    if raw_input("Want to train the server from scratch ? (y/n) ") == "y":
        cloudMessage = requests.post(url+'freshtrain',data=post_fields, allow_redirects=True)
    else:
        cloudMessage = requests.post(url+'train',data=post_fields, allow_redirects=True)
    
    if cloudMessage=='OK':
        print('Cloud has been trained, starting inference')
    else:
        print('There was an error training the cloud...')

ct = 0
while True:
    with Lepton() as l:
        cap,_ = l.capture()

    cv2.normalize(cap, cap, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(cap, 8, cap)
    cap = np.uint8(cap)
    height, width = cap.shape[:2]
    
    cap = cv2.flip( cap, -1 )
    
    enc = basic_encrypt_scramble(cap)
    gray = io.BytesIO()
    gray.write(cv2.imencode('.png', enc)[1].tostring())
    post_img = base64.b64encode(gray.getvalue())

    post_fields = dict(image = post_img)
    gray.seek(0)
    gray.truncate(0)

    message = ''

    try:
        cloudMessage = requests.post(url+'infer',data=post_fields, allow_redirects=True)
        message = cloudMessage.content
        
    except:
        None

    cap = cv2.resize(cap, (8*width, 8*height), interpolation = cv2.INTER_CUBIC)
    if message!='':
        cv2.putText(cap,message, (5,40), cv2.FONT_HERSHEY_COMPLEX, 2, 255)

    cv2.imshow('frame',cap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ct+=1

cv2.destroyAllWindows()
