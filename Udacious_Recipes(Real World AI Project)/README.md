# Udacious Recipes
This repository contains the showcase project developed for the Secure and Private AI Scholarship Challenge. For more details, please visit this site

## The Team
| Name | Slack Handle |
| ------ | ------ |
| Biswajit Banerjee | @Biswajit Banerjee |
| Aisha Javed | @Aisha Javed |
| Ebinbin Ajagun | @Ebinbin Ajagun |
| Evangelia Giannakou | @Evi |
| Oudarjya Sen Sarma | @Oudarjya Sen Sarma|
| Md. Mahedi Hasan Riday | @Mahedi|

## Here's What Udacious Recipe Does For You!

**Simply enter the url you want to generate a recipe from in our [Recipes Webpage](https://evigian.github.io/udacious-recipes/) & below is what Udacious Recipe will do for you:
It will automatically grab all the text on that webpage, feed it into our Intelligent Recipe Bot, and Drill out all 
the recipe relevant information only as an output with all clutter removed. You will be presented with Recipe Name, Directions, Ingredients.
Furthermore, Ingredient Name, Quantity & Unit extracted too! Everything Neat & Crisp!**

## A sneakpeak of what's happening under the hood
### Building the Udacious Recipe ..... The Pipeline ....
1. Collection of recipes from various recipes websites.
2. Preprocessing of the given recipes which includes data cleaning and obtaining word embeddings. 

3. Training Text Classification Model on the preprocessed text to classify a given piece of text into one of the following 4 categories:
- Recipe Name
- Recipe Direction
- Recipe Ingredient
- Recipe Junk - Not Recipe Relevant Information

4. Training Named Entity Recognition Model (NER) to extract Ingredient Name, Quantity & Unit from a given piece of text if it is classified as a Recipe Ingredient Text.

## Current Proceedings Include 
- Data Collection of Recipes from various sources 
- Data Preprocessing 

## Requirements
* Python 3.7 or above
* PyTorch 1.1.0 or above
* PySyft 0.1 or above (optional)
* Matplotlib 3.0 or above (optional)





