# This is the repo for #sg_project-t-shirt challenge :shirt:


<img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/simpleicons.svg" /> <ins> OVERVIEW</ins></br>
**Project T-Shirt** is a design competition initiative created by fellow scholars of [Secure and Private AI Challenge](https://www.udacity.com/facebook-AI-scholarship) under the sponsorship of Udacity and Facebook. The main goal of Project T-shirt is:
1.	To offer students with varying levels of AI experience an opportunity to complete a project.
2.	To create a souvenir of the experience. Top three voted images will be featured on a popular swags such as t-shirts, mugs, hoodies, stickers and more. Later on  everyone will be able to purchase those merchants from official [TeeSpring website](https://teespring.com/ ). It’s a great initiative as all proceeds go to charity. </br></br>

The bar was high and all participants worked very hard to create high quality images. Each picture was special and told unique story. Total of :three: :three: :eight: images were submitted. Our team UNSTOPPABLE including [@agatagruza](https://github.com/agatagruza), [@amalphonse](https://github.com/amalphonse) and [@esridhar126](https://github.com/esridhar126) uploaded :eight: :seven: pictures and at the end :star2::star2::star2:  **won the challenge**  :star2::star2::star2: </br></br>

:shirt: WINNING PICTURE: :shirt:
![WINNER](https://user-images.githubusercontent.com/7014697/63242448-42f10a80-c20c-11e9-9ab5-ab856cdb8f3c.jpg)</br>


<img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/simpleicons.svg" /> <ins> NEURAL STYLE TRANSFER WITH OPEN CV</ins></br>
We were asked to generate an image using a ML/AI method. Based on available resources, our knowledge and experience we have decided to use Neural style transfer with Open CV. The original neural style transfer algorithm was introduced by [Gatys et al. in their 2015 paper, A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf). The same algorithm was applied here. </br>
As a picture is worth a thousand words, below you will find an explanation of how does Neural Style Transfer with OpenCV work:
<p align="center">
  <img src="https://user-images.githubusercontent.com/7014697/63244674-e7764b00-c212-11e9-91da-1931a55e26d8.jpg">
</p>

In summary, neural style transfer is an optimization technique used to take two images, a content image and a style image and blend them together and produce a "painted" like image as output.


Neural style transfer uses three types of loss to merge the images
content loss, style loss and total variation loss. The image is first trained to reproduce the style and then the network is trained to apply the image content. Training the model uses instance normalization than batch normalization for faster real time performance and also for aesthetically pleasing images.

For our project we decided to chose this model because it provides fast results and requires less epochs to train.</br></br>


<img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/simpleicons.svg" /> <ins> CODE IMPLEMENTATION</ins></br>
1) Upload two images, one for content and another for style.
2) Take the style from one image using calculated style loss.
3) From the content image take the content using content loss. 
4) Merge the style with the content.</br></br>


<img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/simpleicons.svg" /> <ins> INSTALATION</ins></br>
In order to replicate above picture you will need to download code</br> 
```git clone https://github.com/agatagruza/project-t-shirt-challenge.git``` and the run command</br> 
```python neural_style_transfer.py --image images/scholar.jpeg --model models/instance_norm/the_scream.t7``` </br>
In mentioned example the Scream style model was in use.</br>

Full implementation of neural style transfer os available [neural_style_transfer.py](https://github.com/agatagruza/project-t-shirt-challenge/blob/neural-style-transfer/neural_style_transfer.py)</br></br>


<img height="32" width="32" src="https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/simpleicons.svg" /> <ins> RESOURCES</ins></br>
:link: https://trello.com/b/ljgxuVOu/gan-t-shirt-contest </br>
:link: https://www.youtube.com/watch?v=8_vhbNpyIk4&feature=youtu.be </br>
:link: https://github.com/ProGamerGov/Torch-Models </br>

### **Contributors:**
:cyclone: **Agata Gruza (contribution 70%)** 
- GitHub: [@agatagruza](https://github.com/agatagruza) 
- Linkedin: [@agatagruza](https://www.linkedin.com/in/agatagruza/)</br>

:cyclone: **Anju Mercian (contribution (30%)**
- GitHub: [@amalphonse](https://github.com/amalphonse)
- Linkedin: [@amalphonse](https://www.linkedin.com/in/anjumercian/)


:movie_camera::movie_camera::movie_camera: [Link to video slide show](https://youtu.be/bM6w7d9jVsw):movie_camera::movie_camera::movie_camera:
