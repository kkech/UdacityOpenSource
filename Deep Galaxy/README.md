<p align="left">
  <img width="270" src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/logo.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/digantamisra98/Deep-Galaxy/master)

# Deep-Galaxy

Computer Vision based project for classification of Galaxies Images and estimating their major chemical composition from color spectrum.

## Project Description: 

Deep Galaxy is a Computer Vision based project hosted as a Web API where an user can input the valid image of any galaxy and the Web API shall classify the Galaxy into it's specific shape-variant class which includes: 
1. E0
2. E3
3. E7 
4. S0
5. Sa
6. Sb
7. Sc
8. SBa
9. SBb
10. SBc 

*E- Elliptical, S - Spiral, SB - Barred Spiral*

<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/Hubble_sequence_photo.png" width="700" height="400" /></div>

*Image Credits - Wikipedia*

The Classifier will also provide additional information regarding the mass/ density of the Galaxy and the Galaxy Name along with it's year of discovery. Furthermore, based on the spectrum of the Image being in either SHO/ SHN/ SHHe*, it will also provide information regarding the max chemical gaseous composition of the galaxy. 

*SHO - Sulphur, Hydrogen, Oxygen
SHN - Sulphur, Hydrogen, Nitrogen
SHHe - Sulphur, Hydrogen, Helium*

## Team Members: 

|Member Name| Slack Handle|
|---|---|
|[Diganta Misra](https://github.com/digantamisra98)| @Diganta|
|[Venkatesh Prasad](https://github.com/ven-k) | @Venkatesh|
|[T.Vishwaak Chandran](https://github.com/vishwaak) | @Xerous|
|[Disha Mendiratta](https://github.com/dishha) | @Disha Mendiratta|
|[Mushrifah Hasan](https://github.com/Mushrifah) | @Mushrifah Hasan|
|[Sourav Kumar](https://github.com/souravs17031999) | @sourav kumar|
|[Arka](https://github.com/Escanor1996) | @Arka|
| [Shivam Raisharma](https://github.com/ShivamSRS) | @Shivam Raisharma|
|[Anshu Trivedi](https://github.com/AnshuTrivedi) | @Anshu Trivedi|

## Project Flow Chart:

<div style="text-align:center"><img src ="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/flow.png"  width="1000"/></div>

## Sample Data:

|Spiral|Elliptical|
|:---:|:---:|
|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/Elliptical/elliptical.gif" width="250" height="250" /></div>|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/Spiral/spiral.gif" width="250" height="250" /></div>|

### Sample Elliptical Galaxy Spectrum Data:

**Galaxy Name - PGC0000212  
Object ID  : 587730775499735086   
Special Object ID : 211330582687252480**

|Green Filter|InfraRed Filter|NearInfraRed Filter|Red Filter| UV Filter| inRGB|
|:---:|:---:|:---:|:---:|:---:|:---:|
|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/PGC0000212/GreenFilter/pgc0000212_greenFilter.png" width="120" height="120" /></div>|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/PGC0000212/InfraredFilter/pgc0000212_infraredFilter.png" width="120" height="120" /></div>|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/PGC0000212/NearInfraredFilter/pgc0000212_nearInfrared.png" width="120" height="120" /></div>|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/PGC0000212/RedFilter/PGC0000212_redFilter.png" width="120" height="120" /></div>|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/PGC0000212/UVFilter/pgc0000212_UVFilter.png" width="120" height="120" /></div>|<div><img src="https://github.com/digantamisra98/Deep-Galaxy/blob/master/Assets/DeepGalaxy/PGC0000212/inRGB/PGC0000212.png" width="120" height="120" /></div>|


