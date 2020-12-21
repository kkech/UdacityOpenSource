#Malaria_Detection_From_Cell_Images

#Project from #sg_spai_health done individually.

[Data Set for the rpoject](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

# Project overview

This is a project I've built on Detecting whether a Human cell is infected with the Malaria Virus or not. Ive use the Data From  [Kaggle Data Set](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

# Approaches followed 

1. Preprocessing- I've added a Gaussian Blur in the images to smoothen the images!!!

2. Then Ive converted the mages to Gray-scale images so that the Contours(of the Malaria Virus) are more visible. 

3. Then I've run a Contour Detection Algorithm on the Processed Images. The strategy is that if only one Contour is detected, that means that the cell is Unaffected, as it is just the Outline of the Cell Image. And if mone that one contour is detected then cell is parasitized. 
I've run my Script twice to get csv files from both the folders i.e. Uninfected & Parasitized!!!

4. Then I've Generated a .csv File from the images after making them pass through the Contour Detection Algorithm. In this .csv file I've shown separately the teh no of contours detected and their areas!!!
here's a screenshot of the CSV File. [Screen shot](https://gyazo.com/f3cca5998c2df533ef3cb7037ebb1863)

5. From the areas & contours we'll get an idea whetheer the cell is infected or not.

<h6>Till now all of this I've done n the malaria.py file.</h6>

6. Now in the malaria_classification.py file I've used a RandomForest Classifier with torch.nn to get a Precision and Recall Report of How Accurate my Predictions. I got an average Precision and Recall of 0.1.

Here's my Output

[LINK](https://gyazo.com/504e93fa939d3527585dd72468e3b5e9)



# Outcome of the project

Used Contour Detection to check whether the cell is Uninfected or Parasitized.

here's a screenshot of the CSV File. [Screen shot](https://gyazo.com/f3cca5998c2df533ef3cb7037ebb1863)


I've trained a RandomForest classifier with torch.nn and got an average Precision and Recall of 0.1

Here's my Output

[LINK](https://gyazo.com/504e93fa939d3527585dd72468e3b5e9)


# Project Participants
 



|Oudarjya Sen Sarma|@Oudarjya Sen Sarma|https://github.com/oudarjya718




### Here's the Workflow of my project!!!

![BlockDiagramLast](https://gyazo.com/b12163104bf4b0db516f0223abe7c99d)

### Use case of my Malaria Dection Project:

- <b> To be used by Microscopists and Doctors</b>
   A very good use case for this project would be deploying  an app for a Microscopist who's is from a remote area with less facilities and he needs to do these detections quickly.


### Using Pytorch :
I used the torch.nn module from Pytorch to train a Random

### Our Future Enhancement(Our Ultimate Goal):
-A very good use case for this project would be deploying  an app for a Microscopist who's is from a remote area with less facilities and he needs to do these detections quickly. S for future enhancements we have thought that we'll convert this project in to an Android App, or a Web app given whatever is more suitable for the User!!!