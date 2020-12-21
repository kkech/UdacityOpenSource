# Real Time Object Detection Using Pytorch 

## INTRODUCTION:
This project is an attempt, to implement Real Time Object Detection using Yolo v3 and OpenCv and Pytorch from scratch. The aim is to create an android app which can be used to detect object in real time with an option to detect specific object from within the set of images or video, and maybe on later stages use the object to reverse search the internet for information and facial recognition.

Most of the code has been inspired from the awesome tutorial by @ayooshkathuria (https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/ ) and the paper by @pjreddie ( https://pjreddie.com/media/files/papers/YOLOv3.pdf ). The code is based on the official code of YOLO v3 ( https://pjreddie.com/darknet/yolo/ ) , as well as a PyTorch port of the original code, by @marvis (https://github.com/marvis/pytorch-yolo2 ).

## WHY CHOOSE YOLO?
YOLOv3 is the latest variant of a popular object detection algorithm YOLO – You Only Look Once, a state-of-the-art, real-time object detection system. The published model recognizes 80 different objects in images and videos, but most importantly it is super fast and nearly as accurate as Single Shot MultiBox (SSD).

Starting with OpenCV 3.4.2, you can easily use YOLOv3 models in your own OpenCV application.

On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev. YOLOv3 is extremely fast and accurate. In mAP measured at .5 IOU YOLOv3 is on par with Focal Loss but about 4x faster. Moreover, you can easily tradeoff between speed and accuracy simply by changing the size of the model, no retraining required! For more information visit this link. (https://pjreddie.com/darknet/yolo/)

![Comparison to Other Detectors](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/2019-08-21%2002_07_25-YOLOv3.pdf.png)

## YOLO v3: Better, not Faster, Stronger
The R-CNN family of techniques use regions to localize the objects within the image. The network does not look at the entire image, only at the parts of the images which have a higher chance of containing an object.

The YOLO framework (You Only Look Once) on the other hand, deals with object detection in a different way. It takes the entire image in a single instance and predicts the bounding box coordinates and class probabilities for these boxes. The biggest advantage of using YOLO is its superb speed – it’s incredibly fast. YOLO also understands generalized object representation.

YOLOv3 came in about April 2018 and it adds further small improvements, included the fact that bounding boxes get predicted at different scales. The underlying meaty part of the network, Darknet, is expanded in this version to have 53 convolutional layers

## Darknet
Darknet is a framework to train neural networks, it is open source and written in C/CUDA and serves as the basis for YOLO. The original repository, by J Redmon (also first author of the YOLO paper), can be found here ( https://github.com/pjreddie/darknet ). Have a look at his website as well (https://pjreddie.com/darknet/ ). Darknet is used as the framework for training YOLO, meaning it sets the architecture of the network.

***For more details on Yolo and darknet read this article by Martina Pugliese (https://martinapugliese.github.io/recognise-objects-yolo/) and Ayoosh Kathuria ( https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b) *** 

## OpenCV
OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception. To know more visit the website (https://opencv.org)

## How does Yolo V3 Works?

Before we begin, What’s the difference between image classification (recognition) and object detection? In classification, you identify what’s the main object in the image and the entire image is classified by a single class. In detection, multiple objects are identified in the image, classified, and a location is also determined (as a bounding box).  

**A Fully Convolutional Neural Network**

YOLO makes use of ***only convolutional layers, making it a fully convolutional network (FCN)***. *It has 75 convolutional layers, with skip connections and upsampling layers. No form of pooling is used, and a convolutional layer with stride 2 is used to downsample the feature maps.* This helps in preventing loss of low-level features often attributed to pooling.

Being a FCN, YOLO is invariant to the size of the input image. However, in practice, we might want to stick to a constant input size due to various problems that only show their heads when we are implementing the algorithm.

A big one amongst these problems is that if we want to process our images in batches (images in batches can be processed in parallel by the GPU, leading to speed boosts), we need to have all images of fixed height and width. This is needed to concatenate multiple images into a large batch (concatenating many PyTorch tensors into one)

The network downsamples the image by a factor called the stride of the network. For example, if the stride of the network is 32, then an input image of size 416 x 416 will yield an output of size 13 x 13. Generally, *stride of any layer in the network is equal to the factor by which the output of the layer is smaller than the input image to the network.*

**Interpreting the output**

Typically, (as is the case for all object detectors) the features learned by the convolutional layers are passed onto a classifier/regressor which makes the detection prediction (coordinates of the bounding boxes, the class label.. etc).

In YOLO, the prediction is done by using a convolutional layer which uses 1 x 1 convolutions.

Now, the first thing to notice is the output is a feature map. Since it has 1 x 1 convolutions, the size of the prediction map is exactly the size of the feature map before it. In YOLO v3 (and it's descendants), the way you interpret this prediction map is that each cell can predict a fixed number of bounding boxes.

    Though the technically correct term to describe a unit in the feature map would be a neuron, calling it a cell makes it more intuitive in our context.

**Depth-wise, we have (B x (5 + C)) entries in the feature map.** B represents the number of bounding boxes each cell can predict. According to the paper, each of these B bounding boxes may specialize in detecting a certain kind of object. Each of the bounding boxes have 5 + C attributes, which describe the center coordinates, the dimensions, the objectness score and C class confidences for each bounding box. YOLO v3 predicts 3 bounding boxes for every cell.

***You expect each cell of the feature map to predict an object through one of it's bounding boxes if the center of the object falls in the receptive field of that cell.*** (Receptive field is the region of the input image visible to the cell. Refer to the link on convolutional neural networks for further clarification).

This has to do with how YOLO is trained, where only one bounding box is responsible for detecting any given object. First, we must ascertain which of the cells this bounding box belongs to.

To do that, we divide the input image into a grid of dimensions equal to that of the final feature map.

Let us consider an example below, where the input image is 416 x 416, and stride of the network is 32. As pointed earlier, the dimensions of the feature map will be 13 x 13. We then divide the input image into 13 x 13 cells.
Then, the cell (on the input image) containing the center of the ground truth box of an object is chosen to be the one responsible for predicting the object. In the image, it is the cell which marked red, which contains the center of the ground truth box (marked yellow).

![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_07_31-Tutorial%20on%20implementing%20YOLO%20v3%20from%20scratch%20in%20PyTorch.png)

Now, the red cell is the 7th cell in the 7th row on the grid. We now assign the 7th cell in the 7th row on the feature map (corresponding cell on the feature map) as the one responsible for detecting the dog.

Now, this cell can predict three bounding boxes. Which one will be assigned to the dog's ground truth label? In order to understand that, we must wrap out head around the concept of anchors.

    Note that the cell we're talking about here is a cell on the prediction feature map. We divide the input image into a grid just to determine which cell of the prediction feature map is responsible for prediction

**Anchor Boxes**

It might make sense to predict the width and the height of the bounding box, but in practice, that leads to unstable gradients during training. Instead, most of the modern object detectors predict log-space transforms, or simply offsets to pre-defined default bounding boxes called anchors.

Then, these transforms are applied to the anchor boxes to obtain the prediction. YOLO v3 has three anchors, which result in prediction of three bounding boxes per cell.

Coming back to our earlier question, the bounding box responsible for detecting the dog will be the one whose anchor has the highest IoU with the ground truth box.

**Making Predictions**

The following formulae describe how the network output is transformed to obtain bounding box predictions.
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_07_49-Tutorial%20on%20implementing%20YOLO%20v3%20from%20scratch%20in%20PyTorch.png)

***YOLO Equations***
bx, by, bw, bh are the x,y center co-ordinates, width and height of our prediction. tx, ty, tw, th is what the network outputs. cx and cy are the top-left co-ordinates of the grid. pw and ph are anchors dimensions for the box.

**Center Coordinates**

Notice we are running our center coordinates prediction through a sigmoid function. This forces the value of the output to be between 0 and 1. Why should this be the case? Bear with me.

Normally, YOLO doesn't predict the absolute coordinates of the bounding box's center. It predicts offsets which are:

    Relative to the top left corner of the grid cell which is predicting the object.

    Normalised by the dimensions of the cell from the feature map, which is, 1.

For example, consider the case of our dog image. If the prediction for center is (0.4, 0.7), then this means that the center lies at (6.4, 6.7) on the 13 x 13 feature map. (Since the top-left co-ordinates of the red cell are (6,6)).

But wait, what happens if the predicted x,y co-ordinates are greater than one, say (1.2, 0.7). This means center lies at (7.2, 6.7). Notice the center now lies in cell just right to our red cell, or the 8th cell in the 7th row. This breaks theory behind YOLO because if we postulate that the red box is responsible for predicting the dog, the center of the dog must lie in the red cell, and not in the one beside it.

Therefore, to remedy this problem, the output is passed through a sigmoid function, which squashes the output in a range from 0 to 1, effectively keeping the center in the grid which is predicting.

**Dimensions of the Bounding Box**

The dimensions of the bounding box are predicted by applying a log-space transform to the output and then multiplying with an anchor.
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_07_58-Tutorial%20on%20implementing%20YOLO%20v3%20from%20scratch%20in%20PyTorch.png)

*How the detector output is transformed to give the final prediction. Image Credits. http://christopher5106.github.io/*

The resultant predictions, bw and bh, are normalised by the height and width of the image. (Training labels are chosen this way). So, if the predictions bx and by for the box containing the dog are (0.3, 0.8), then the actual width and height on 13 x 13 feature map is (13 x 0.3, 13 x 0.8).

**Objectness Score**

Object score represents the probability that an object is contained inside a bounding box. It should be nearly 1 for the red and the neighboring grids, whereas almost 0 for, say, the grid at the corners.

The objectness score is also passed through a sigmoid, as it is to be interpreted as a probability.

**Class Confidences**

Class confidences represent the probabilities of the detected object belonging to a particular class (Dog, cat, banana, car etc). Before v3, YOLO used to softmax the class scores.

However, that design choice has been dropped in v3, and authors have opted for using sigmoid instead. The reason is that Softmaxing class scores assume that the classes are mutually exclusive. In simple words, if an object belongs to one class, then it's guaranteed it cannot belong to another class. This is true for COCO database on which we will base our detector.

However, this assumptions may not hold when we have classes like Women and Person. This is the reason that authors have steered clear of using a Softmax activation.

**Prediction across different scales.**

YOLO v3 makes prediction across 3 different scales. The detection layer is used make detection at feature maps of three different sizes, having strides 32, 16, 8 respectively. This means, with an input of 416 x 416, we make detections on scales 13 x 13, 26 x 26 and 52 x 52.

The network downsamples the input image until the first detection layer, where a detection is made using feature maps of a layer with stride 32. Further, layers are upsampled by a factor of 2 and concatenated with feature maps of a previous layers having identical feature map sizes. Another detection is now made at layer with stride 16. The same upsampling procedure is repeated, and a final detection is made at the layer of stride 8.

At each scale, each cell predicts 3 bounding boxes using 3 anchors, making the total number of anchors used 9. (The anchors are different for different scales)
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_08_08-Tutorial%20on%20implementing%20YOLO%20v3%20from%20scratch%20in%20PyTorch.png)

The authors report that this helps YOLO v3 get better at detecting small objects, a frequent complaint with the earlier versions of YOLO. Upsampling can help the network learn fine-grained features which are instrumental for detecting small objects.

**Output Processing**

For an image of size 416 x 416, YOLO predicts ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10647 bounding boxes. However, in case of our image, there's only one object, a dog. How do we reduce the detections from 10647 to 1?

**Thresholding by Object Confidence**

First, we filter boxes based on their objectness score. Generally, boxes having scores below a threshold are ignored.

**Non-maximum Suppression**

NMS intends to cure the problem of multiple detections of the same image. For example, all the 3 bounding boxes of the red grid cell may detect a box or the adjacent cells may detect the same object.
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_08_21-Tutorial%20on%20implementing%20YOLO%20v3%20from%20scratch%20in%20PyTorch.png)

***credit for the above explanation goes to Ayoosh Kathuria in his tutorial (https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)***

## Completed task based outputs
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_20_52-eriklindernoren_PyTorch-YOLOv3_%20Minimal%20PyTorch%20implementation%20of%20YOLOv3.png)
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_20_40-eriklindernoren_PyTorch-YOLOv3_%20Minimal%20PyTorch%20implementation%20of%20YOLOv3.png)
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_20_27-eriklindernoren_PyTorch-YOLOv3_%20Minimal%20PyTorch%20implementation%20of%20YOLOv3.png)

## End Result
After deploying it into android app it should look something like this.

![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_29_16-GitHub%20-%20ultralytics_yolov3_%20YOLOv3%20in%20PyTorch%20_%20ONNX%20_%20CoreML%20_%20iOS.png)
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/2019-08-21%2003_29_25-GitHub%20-%20ultralytics_yolov3_%20YOLOv3%20in%20PyTorch%20_%20ONNX%20_%20CoreML%20_%20iOS.png)
![](https://github.com/shaistha24/UdacityOpenSource/blob/Shaistha/Shaistha/real_time_object_detection_using_pytorch/restmb_idxmake.php.jpg)

*photo credits for android view goes to https://github.com/ultralytics/yolov3 )*

## Future Updates
- Add it to android
- Adding Facial recognition feature
- Detect object from the set of given images or videos.
- Try to implement and test run on different datasets.

## Social Impact
This app is mainly designed for research purpose and for learning. The featured audience is any developer interested in Computer Vision and also for AI Security/Data researchers as this can be used to find the images oor video time for specific object from the pile of images and videos!

