# Project-Showcase-Challenge: A Federated approach for Diabetic Retinopathy diagnosis 
##### Udacity and Facebook Private and Secure AI Challenge

### Developers:
- ThienAn
     * Slack user: ThienAn
     * Github: thienan092
- José Luis Samper (jluis.samper@gmail.com)
     * Slack user: J. Luis Samper
     * Github: JL-Samper

# Abstract
Retinal affections is one of the major causes of blindness in the population. Early detection and appropiate treatment are crucial to stop the progress of the illness. Diagnoses relies on the expertise and visual acuteness of the ophthalmologist to detect retinal symptoms [1]. The oculist checks the retina of both eyes by using special magnifying glasses and scan images, which alongside the patient's records provide information enough to detect the illness in most of the cases. However, a part of the population is highly exposed to late detection either by inability to afford the medical examination, lack of awareness or retina experts high-occupation. 

Among retinal affections, Diabetic Retinopathy (DR) has a high incidence rate [2]. During the last decade, big efforts have been done to obtain machine learning models able to detect the illness and diagnose it, like the population of journal articles and competitions denote. A successfull and useful diagnosis system should not only have a high accuracy, but also deliver scalability, data safety and usability. In this regard, a multiplatform application for retinal imaging diagnoses that applies federated learning methods can have a real impact and migration to real scenearios where medical personnel and patients interact with it.  

In this document we present a federated learning application that serves from state of the art methods for image recognition and classification to diagnose diabetic retinopathy. The system detects five different states of the retina and can be embedded in different devices, from computers to embedded boards. Currently, we are focusing on raspberry pi 4, and pc, but the future prospect also includes migration to mobile devices and other boards.

# Introduction

The last report of the World Health Organization estimate 36 million blind people worldwide, out of which 81% is avoidable and related to retinopathy [1]. A retinopathy occurs when blood leaks from the vessels to the retina creating sight problems which include distorted vision, dark zones in sight and color impairment among others. The retina is a membrane at the back of the eye sensitive to light, which is irrigated by blood vessels and moves over vitreus [2].

Diabetic retinopathy (DR) is among the most-common retinal affections with a 31,2% incidence rate in diabetic people [1].  One of the measures to reduce DR risks consists on regular screening [2]. In these screening images, experts look for signs of fluid entrance (like hollows or dark zones), aneurysms, exudates and excessive capilarity in the retina [3]. The National Eye Institute distinguishes between four stages of diabetic retinopathy: mild nonproliferative, moderate nonproliferative, severe nonproliferative and proliferative. The former two are characterised by retina distortion and the existance of balloon-like shapes, whereas in the latter two a proliferation of blood vessels occurs [4]. 

![alt text](https://media.discordapp.net/attachments/602098962719309856/602109554930614273/nrdp201612-f4.png)

So as to enhance diagnosis and preventive cares, several studies and challenges of machine learning models for DR detection have populated during the last decades. Some authors advocate for preprocessing and manual design of detectors. In these studies, several processing techniques are applied to highligth the retinal symptoms of retinopathy which are then classified by applying different types of functions like the feature candidates proposed by Flemming [5]. Habib et al [6] have recently analysed 70 features of retinal images and their impact on decision candidates performance. The retinal images are preprocessed by applying salt and pepper methods, surf and a vessel removal technique, developed by the authors. Then a decision tree classifies each of the symptoms detected by analysing the response of a series of weak classifiers. 

Attempts to improve performance and diagnosis have eventually led to the usage of deep learning models too [7]. Simple convolutional networks with few layers can stage diabetic retinopathy with an accuracy around 86,9% [8]. It is to expect than an increase on the depth of the model and the training database size can boost performance up. The American Academy of Ophthalmology have reported 93% accuracy for macular degeneration diagnosis using deep convolutiona networks based on VGG16 [9]. The macula is the part of the retina responsible for central vision which focuses on objects and acquires fine details [10]. 

Similar results have already been obtained for glaucoma. Glaucoma is an affection of the optical nerve whose diagnoses can only be done by medical eye experts. Its detection is really complex due to the fundus variance and shape of retina symptoms [10]. In this context, convolutional models with six or less layers reach accuracies around 83% [11], whereas deeper models increase performance to 93% [12]. 

In consequence, there is a great interest in obtaining neural models for retinal disease diagnosis and the results are promising. Deep Learning and technological development allow designing complexer models that push performance forward. However, there are important topics that have to be addressed in order to migrate these systems to society which can hamper the adoption of the technology by the big public. These topics are related with real-case implementation, user's privacy and safety. To our knowledge, there is no specific work already done to approach a federated learning model that diagnoses retinal affections while preserves user's privacy. In this project we create a federated learning architecture with a pc and a raspberry pi 4 that hosts several workers to diagnose diabetic retinopathy. Additionally, different neural models have been evaluated to boost performance which currently include: simple convolutional networks (2 to six layers), high-depth convolutional models and GANs.   The system are trained and tested using Kaggle competition databases for DR detection from 2015 and 2019.

# System Overview

The system can be mainly divided in two parts: the neural network and the federated environment. The former covers the design of the neural algorithm and its performance for DR diagnosis using [kaggle datasets](https://www.kaggle.com/benjaminwarner/resized-2015-2019-blindness-detection-images), whereas the latter covers the integration of the neural network in a federated architecture. The whole development have been done with PyTorch and PySyft. 

#### Neural Network Implementation

infoGAN[13] and transfer learning with VGG16[12] are the neural networks currently in use after evaluating its performance alongside a 2-layer convnet and transfer learning with densenet121. The dataset provides two images of the retina of each eye (that's important since natural eye lessions commonly appear in both eyes which can be evaluated in future improvements). There are five labels or classes to diagnose the retina:
- No retinopathy
- Mild nonproliferative retinopathy
- Moderate nonproliferative retinopathy
- Severe nonproliferative retinopathy
- Proliferative retinopathy

The dataset is not evenly-representative of these labels. As the following chart illustrates, healthy eyes are predominant:
![alt_text](https://cdn.discordapp.com/attachments/602098962719309856/604644893884940289/unknown.png)

To avoid bias and overfitting towards healthy eyes, we have downsized the dataset to have an evenly-distributed population of samples. Additionally, images are preprocessed to avoid format-related bias [14] by applying crop and normalization methods. 
The accuracy obtained is around 89%.

Some authors like Fleming[5] and Habib[6] have reported the benefits in assembled classification when applying vessel removal and an exhaustive preprocessing to highlight the retinal symptons under study. In our opinion, adding a red-color discriminant, fundus correction and vessel extraction can boost accuracy since the result will provide a simplified image with the imperfections of the retina. We will also have to deal with the induced artifacts resulting from the preprocessing algorithm as other authors have already stated [6]. We are currently working on this implementations which we plan to carry out with OpenCV and PyTorch. 

#### Federated Learning

The system proposed can be used by medical personnel as assistance and patients that are under risk given their medical record. We have also considered the usage in third-world countries and communities without access to retinal experts. Therefore, we need a secure server and at least one server of the model trained model. This server communicates with client workers that can be around the world in different devices. All the image data and parameters are considered sensitive information that could be hacked to obtain medical records from individual, in consequence, we are using a privacy worker for encryption.

To approach such an architecture we have 2 PCs, one running Ubuntu Bionic and another running OS X Mojave and a Raspberry Pi 4. We also have a Raspberry Pi Zero which will be added to the architecture later. We would like to evaluate performance and adaptability of the system to different specs since it can have a huge impact on commercialisation and adaption to poor countries. 

The federated learning environment is done with PySyft following OpenMined guidelines for websockets. After troubleshooting some issues with raspberry pi 4 installation and power sortages we have been able to run a simple example for MNIST dataset using websockets between OSX and RPi4. We are starting the implementation of the transfer-learning neural algorihtm, explained in previous subsection, in this scheme. 

![alt_text](https://media.discordapp.net/attachments/602098962719309856/613348507356626957/unknown.png)

# Troubleshooting and guides

During Raspberry Pi 4 set up with PySyft and Pytorch some issues have been found. We have based on [OpenMined Tutorial](https://blog.openmined.org/federated-learning-of-a-rnn-on-raspberry-pis/) to set RPi 3B+ for federated RNN. Issues and troubleshooting that we have faced can be sum up as follows:

- Pytorch and Syft wheels in pip3 work properly but you need to pay attention to the version installed. The best options is forcing the installation of the version you need to this end check [syft requirements](https://github.com/OpenMined/PySyft). If you don't force the installation you may end with a torch version like "vX.X.XX.post2" which will create several issues in further wheels installations. 
- Upgrade pip version installed with raspbian so new dependencies and wheels can be found.
- Avoid screen blanking by installing xscreensaver and disabling screensaver (Preferences --> screensaver)
- Force hdmi output to never blank by adding 
    > hdmi-safe = 1 
    > in /boot/config.txt. 
- It is [also suggested](https://www.raspberrypi.org/forums/viewtopic.php?t=139538) to change the lightDM configuration to disable blanking by adding: 
    > xserver-command=X -s 0 -dpms 
    > to /etc/lightdm/lightdm.conf the file. 
- If your Rpi randomly shuts down or keeps blanking screen after applying previous modifications, you may have a power supply issue either by hdmi faulty cable or power adapter failure. 
- Bear in mind that there is a connection issue at the power connector of Pi4 that can produce overheating and faulty events if the power source is not the one specified by the manufacturer as [stated in different sources](https://hackaday.com/2019/07/16/exploring-the-raspberry-pi-4-usb-c-issue-in-depth/). 
- If the board keeps failing, try to replace it for a new one you may have a defectuous one. Many posts have recently appear in this regard in [forums](https://hackaday.com/2019/06/28/power-to-the-pi-4-some-chargers-may-not-make-the-grade/).

# References
[1] Venkata SM Gudlavalleti  et al. "Public health system integration of avoidable blindness screening and management, India". Bulletin of the World Health Organization . Aug 2018. [Online] https://www.who.int/bulletin/volumes/96/10/18-212167/en/

[2] Retinopathy. Harvard Health Publishing. Harvard Medical School. [Online] https://www.health.harvard.edu/a_to_z/retinopathy-a-to-z

[3] Christian Nordqvist. "Diabetic retinopathy: Causes, symptoms, and treatments". Medical News Today. August 2017. [Online] 

[4] "Facts About Diabetic Eye Disease". National Eye Institute. [Online] https://nei.nih.gov/health/diabetic/retinopathy

[5] Alan D. Fleming et al. "Automated Microaneurysm Detection Using Local Contrast Normalization and Local Vessel Detection". IEEE TRANSACTIONS ON MEDICAL IMAGING, Vol. 25, No. 9, Sept 2006. 

[6] Habib et al. "Detection of microaneurysms in retinal images using an ensemble classifier". Informatics in Medicine Unlocked. 2017. 

[7] Annahita Forghan et al. "Can AI Detect Diabetic Retinopathy More Accurately?". Diabetes in Control. Jan 2019. [Online] http://www.diabetesincontrol.com/can-ai-detect-diabetic-retinopathy-more-accurately/

[8] Mohamed Shaban et al. "Automated staging of Diabetic Retinopathy Using a 2D Convolutional Neural Network". AMIA Jt Summits Transl Sci Proc. 2018; 2018: p147–155. 

[9] Cecilia S. Lee et al. "Deep Learning Is Effective for Classifying Normal versus Age-Related Macular Degeneration OCT Images". American Academy of Ophthalmology. 2016.

[10] Age-Related Macular Degeneration (AMD). Columbia University Department of Ophthalmology. [Online] https://www.columbiaeye.org/eye-library/age-related-macular-degeneration.

[11] Xiangyu Chen et al. "Glaucoma detection based on deep convolutional neural network".  2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). Aug 2015.

[12] Manal Al Ghamdi et al. " Semi-supervised Transfer Learning for Convolutional Neural Networks for Glaucoma Detection".  ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). May 2019.  

[13] Jefferson L. P. Lima et al. "Heartbeat Anomaly Detectionusing Adversarial Oversampling". arXiv preprint arXiv:1901.09972.

[14] Tom Aindow. "Be careful what you train on. Kaggle APTOS competitions". [Online] https://www.kaggle.com/taindow/be-careful-what-you-train-on
