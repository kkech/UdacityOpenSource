import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

import collections
import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from utils.convex_hull import CHull




def draw_features():
    # TO DO
    pass

def open_image(image_path):
    
    """
    image_path: string 
        path to image
    """

    try:
        input_img = io.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        

    return input_img
def open_image_cv2(image_path):
    """
    image_path: string 
        path to image
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError("File {} not found.".format(image_path))
    img = cv2.imread(image_path, -1) # Open images unchanged (with transparency)

    return img

def draw_sprite( frame, sprite, x_offset, y_offset):
    """
    frame: array or image
    
    sprite: array or image

    x_offset: integer
        number of colums where put sprite.
    
    y_offset: integer
        number of rows where put sprite.

    """
    h,w = sprite.shape[0], sprite.shape[1]

    imgH, imgW = frame.shape[0], frame.shape[1]

    if y_offset +  h  >= imgH:  #if sprite gets out of image in the bottom
        sprite = sprite[0: imgH - y_offset, :, :]

    if x_offset + w >= imgW: 
        #if sprite gets out of image in the bottom
        sprite =sprite[:,0: imgW - x_offset , :]
    if x_offset < 0 : #if sprite gets out of image to the left
        sprite = sprite[ :, abs(x_offset)::,: ]
        w = sprite.shape[0]
        x_offset = 0 
    # For each RGB channel
    for c in range(3):
        # Chanel 4 is for alpha: 255 100% opaque, 0 is transparent.
        alpha = sprite[: , :, 3 ] / 255 # Alpha values are in [0,1]
        frame[y_offset: y_offset + h, x_offset: x_offset + w, c ] = sprite[:,:,c]  * alpha  +  frame[y_offset: y_offset + h , x_offset: x_offset + w, c ] * (1 - alpha)

    return frame 

def adjust_sprite2head(sprite, head_width, head_y_pos, ontop = True):
    """
    sprite: array or image like object
    head_width: int
        head width in pixels
    head_y_pos: int
        vertical position of head

    return sprite, y_orig 
    rescaled sprite and vertical position of sprite 
    """

    (h_sprite, w_sprite) = (sprite.shape[0],sprite.shape[1])

    #print("Head width {} width sprite {}".format(head_width, w_sprite))
    factor = 1.1 *  head_width / w_sprite
    #print("Factor ", factor)
    
    # Resize sprite. Adjust to have the same width as head
    sprite = cv2.resize(sprite, (0,0),fx = factor, fy = factor )

    h_sprite, w_sprite = (sprite.shape[0], sprite.shape[1])

    # Adjust the position of sprite to end where the head begins
    y_orig = head_y_pos  - h_sprite if ontop else  head_y_pos

    # Check if the head is not to close to the top of the image and the sprite would not fit in the screen
    if y_orig < 0:
        sprite = sprite[abs(y_orig)::,:,:] #in that case, we cut the sprite
        y_orig = 0 #the sprite then begins at the top of the image
    
    return sprite, y_orig


def apply_sprite(image, sprite,w,x,y, angle, ontop = True):
    """
    image: array or image like object
    sprite: array or image like object
       
    w:int
    x:int
    y:int


    """
    sprite = rotate_image(img = sprite, angle = angle, scale = 1.0)
    
    sprite, y_final = adjust_sprite2head( sprite , w, y, ontop )
    image = draw_sprite(image,sprite, x, y_final)
    return image





def compute_inclination(point1, point2):
    """
    point1: 2D tuple, array or list 

    point2: 2D  tuple, array or list
    Returns angle between points in degrees
    """

    #find vector 
    vec  = [point2[0] - point1[0]  , point2[1] - point1[1] ]

    angle = vec[0] /  np.sqrt( vec[0]**2  + vec[1] ** 2  )

    return angle 


def calculate_boundbox(list_coordinates):
    """
    coordinates are inverted  x: colums y :rows


    list_coordinates: list of the form [ [x2,y1], [x2,y2], . . .    ]

    returns top  left point (x,y) and width (w) and heigth (h) of rectangle 

    """
  
    x = int(min(list_coordinates[:,0]))
    y = int(min(list_coordinates[:,1]))

    w  =int(max(list_coordinates[:,0]) - x)
    h = int(max(list_coordinates[:,1]) - y)

  
    return x,y,w,h


def get_face_boundbox(points,face_part):
    """
    points: list
        list of lists [ [x1,y1], [x2,y2], ...  ]
     
    face_part: string
        name of feature. Example eye1, eye2.

    return a list of tuple (x,y,w,h) 

    """
    #print("Points", points )

    part_points = get_points_from_feature(name = face_part,preds = points)

   

    x,y,w,h = calculate_boundbox( part_points)


    return x,y,w,h 



def get_best_scaling(target_width, filter_width ):
    """
    target_width: integer
        width for feature in face. For example width of  bounding box for eyes.
    filter_width: integer
        width of filter    
    """
    # Scale width by 1.1

    return 1.1 * (target_width / filter_width)



def get_eye_angle(left_eye,right_eye):
    """
    left_eye: 2D tuple, array or list 

    right_eye: 2D  tuple, array or list
    """

    #find vector 
    vec  = [right_eye[0] - left_eye[0]  , right_eye[1] - left_eye[1] ]

    angle = vec[0] /  np.sqrt( vec[0]**2  + vec[1] ** 2  )

    return angle 



def rotate_image(img, angle, scale):
    """
    img: numpy array or image like object
    angle: int 
        Angle in degrees
    scale: float 
        scale on range [0,1]
    """
    height, width  = (img.shape[0], img.shape[1])
    shape = (width,height)
    center = width / 2, height/2

    rotation_matrix = cv2.getRotationMatrix2D(center,
                                              angle,
                                              scale = 1.0)

    new_img = cv2.warpAffine(img, rotation_matrix, shape,
                             flags = cv2.INTER_LINEAR,
                             borderMode = cv2.BORDER_TRANSPARENT)
    return new_img



def get_points_from_feature(name,preds):
    """
    name: string
        name of feature. Example eye1, eye2.

    preds: list
        list of lists [ [x1,y1], [x2,y2], ...  ]
    return a list of pairs (x,y) 
    """

    points = preds[PRED_TYPES[name].slice]

    return points


def get_eyes_width_heigth(preds):
    eye1, eye2 = (preds[PRED_TYPES['eye1'].slice,0],preds[PRED_TYPES['eye1'].slice,1]),(preds[PRED_TYPES['eye2'].slice,0],preds[PRED_TYPES['eye1'].slice,1])

    #sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)

    y_coordinates = eye1[1] + eye2[1]
    max_y,min_y = np.max(y_coordinates), np.min(y_coordinates) 
    eyes_heigth= int(  max_y  -  min_y  )

    x_coordinates =  eye1[0] + eye2[0]
    max_x,min_x =np.max(x_coordinates) , np.min(x_coordinates)
    eyes_width = int( max_x - min_x   ) 

    return eyes_width, eyes_heigth




# Main Function to apply filters
def add_filter(img_path, filter_name):
    

    img = open_image_cv2(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces, preds = get_features_from_image(img)
    #print("PREDS",preds)


    if len(faces)==0:
        print("No faces Detected in the image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    # Coordinates are inverted  x: colums y :rows

    for x,y,x2,y2,_ in faces:   
       
        
        x,y,x2,y2 = tuple(map(int, [x,y,x2,y2] ))
        # add bounding box to color image
        #cv2.rectangle(img,(x,y),(x2,y2),(255,0,0),2)

        gray_face = gray[y:y2, x:x2]
        color_face = img[y:y2, x: x2]

        # Show Face
        #plt.imshow(color_face)
        #plt.show()

        eyebrow1, eyebrow2 = get_points_from_feature(name ='eyebrow1' ,preds=preds), get_points_from_feature(name = 'eyebrow2', preds = preds)
        
      

        angle =-1* compute_inclination(eyebrow1[0],eyebrow2[0] ) #inclination based on eyebrowns
        

        mouth = get_points_from_feature(name = 'lips', preds = preds)

        # Condition to see if mouth is open
        is_mouth_open = abs(max(mouth[:,0]) -  min(mouth[:,0]) ) >=  20
       


        # Compute width
        w =  int(  y2 - y    ) 
        

        if filter_name == 'hat':
            # Hat
            hat_image = SPRITES['hat']
            
            apply_sprite(img,hat_image ,w,x,y,angle)


        if filter_name == 'mustache':

            x_1, y_1, w_1, h_1 = get_face_boundbox(points = preds  , face_part = 'lips')
            mustache_image = SPRITES['mustache']
            apply_sprite(img,mustache_image,w_1,x_1,y_1,angle )
        
        if filter_name == 'glasses':
            x_3, y_3,_ ,h_3 = get_face_boundbox(points = preds, face_part = 'eyebrow1')
            w_eyes,h_eyes = get_eyes_width_heigth(preds)
            print("W EYES ",w_eyes,w )

            glasses_image = SPRITES['glasses']
            apply_sprite(img, glasses_image,w,x,y_3, angle, ontop= False )

        
        if filter_name == 'flies':
            #to make the "animation" we read each time a different image of that folder
            # the images are placed in the correct order to give the animation impresion
            
            apply_sprite(img, FLIES[list(FLIES.keys())[0]], w , x , y, angle)
            #i+=1
            #i = 0 if i >= len(flies) else i #when done with all images of that folder, begin again

            
        x0, y0, w0, h0 = get_face_boundbox(preds, face_part = "lips")

        # CV2 Assumes (x,y) : (row , column)
        #cv2.rectangle(img,(x0,y0),(x0 + w0,y0  + h0),(255,0,0),1)

        if filter_name == "doggy":

            x3 , y3, w3, h3 = get_face_boundbox(preds, 'nostril')
       
            

            apply_sprite(img,SPRITES['doggy_nose'],w3, x3, y3, angle, ontop = False )

            apply_sprite(img,SPRITES['doggy_ears'], w, x, y , angle )


            if is_mouth_open:
                apply_sprite(img, SPRITES['doggy_tongue'], w0,x0,y0,angle, ontop = False)
        
        if filter_name == 'rainbow':
            
            if True:
                apply_sprite(img, SPRITES['rainbow'], w0,x0,y0,angle, ontop = False)





        #print("Eye 1 {}, Eye 2 {}".format(eye1,eye2))


        #print("Sunglasses width {} and height {} ".format(eyes_width, eyes_heigth )) 

        #sunglasses_resized = cv2.resize(sunglasses, (eyes_width * 2,eyes_heigth * 2),interpolation = cv2.INTER_CUBIC)
        

        #left_hull = CHull(get_points_from_feature('eye1',preds))
        #right_hull = CHull(get_points_from_feature('eye2',preds)) 

        #left_eye_centrum  = left_hull.centrum()
        #right_eye_centrum = right_hull.centrum()
        #eyes_angle = get_eye_angle(left_eye_centrum, right_eye_centrum)

        #print("Eyes angle", eyes_angle)
        # Rotate  sunglasses by angle between eyes
        #sunglasses_final = rotate_image(sunglasses_resized,-eyes_angle, 1 )
        
        # Get anchor for sunglasses
        #anchor_x, anchor_y = int(np.min(eye1[0])),int(np.min(eye1[1]))
     
        
        #Overlay sunglasses over img
        #img = draw_sprite(frame = img , sprite = sunglasses_final, x_offset= anchor_x, y_offset = anchor_y  )
        

        #for point in preds:
        #    cv2.circle(img, tuple(point), 1, (255,255,255), 1)
        
        # Test anchor point for sunglasses
        #cv2.circle(img, (x_,y_), 2, (255,0,0),2)
    #cv2.imshow("Filters", img)


       
    #Change to RGB
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    return result
    
  

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
PRED_TYPES = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
}

"""
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('3d_landmarks_alone.png', bbox_inches=extent)
"""

def get_features_from_image(image):
    """
    image: array 
        image object .
    return boxes_faces (bounding boxes for faces), predictions (facial landmarks)
    """

   
    # Run the 2D face alignment on a test image, with CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True)
    #bounding boxes of faces found in image

    boxes_faces = fa.face_detector.detect_from_image(image[..., ::-1].copy())
    preds = fa.get_landmarks_from_image(image,detected_faces=boxes_faces)[-1]
    
   
    return boxes_faces, preds




def image_2d_landmarks(image_path):
    """
    image_path: string
        path to image.

    """

    try:
        input_img = io.imread(image_path)
    except FileNotFoundError:
        print("File {} not found.".format(image_path))
        

    boxes_faces,preds= get_features_from_image(input_img )

    print("Boxes_faces type", type(boxes_faces), boxes_faces,boxes_faces[0].shape)



    # 2D-Plot
    plot_style = dict(marker='o',
    markersize=4,
    linestyle='-',
    lw=2)

    fig = plt.figure(
                     #figsize = (10.24,10.24)
                     figsize=plt.figaspect(.5)
                    )
    ax = fig.add_subplot(1,1,1) #(1, 2, 1)
    ax.imshow(input_img)

    for pred_type in PRED_TYPES.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    fig.savefig('2d_landmarks.png')








def image_3d_landmarks():
    pass






if __name__ == "__main__":

    filters = ['images/sunglasses.png', 'images/sunglasses_2.png', 'images/sunglasses_3.jpg', 'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png','images/pixel.png']
    filterIndex = 6

    SPRITES = [0,0,0,0,0] #hat, mustache, flies, glasses, doggy -> 1 is visible, 0 is not visible
    
    # Preload sprites 
    sprites_dir = "sprites"
    sprites_paths = os.listdir(sprites_dir)
    sprites_paths = [ p  for p in sprites_paths if os.path.isfile( os.path.join(sprites_dir, p) )  == True ]

    
    SPRITES = {}

    for filename in sprites_paths:
        sprite_path = os.path.join(sprites_dir, filename)
        sprite = open_image_cv2(sprite_path)
        sprite_name = filename[:-4]
        SPRITES[sprite_name] = sprite

        print("sprite_name {} and shape {}".format(sprite_name, sprite.shape))

    FLIES = {}
    flies_dir  ='./sprites/flies'
    flies_paths = os.listdir( flies_dir)
    for filename in flies_paths:
        sprite_path = os.path.join(flies_dir, filename)
        print("Sprite path", sprite_path)
        sprite = open_image_cv2(sprite_path)
        sprite_name = filename[:-4]
        FLIES[sprite_name] = sprite
        print("sprite_name {} and shape {}".format(sprite_name, sprite.shape))

    
    file_path = 'john_wick.jpg'
   
    result = add_filter(file_path,'rainbow')

    plt.axis('off')
    plt.imshow(result)
    plt.show()
    