# SIIM-ACR Pneumothorax Segmentation
[Madison Estabrook](https://github.com/madisonestabrook)
# About
This project serves 2 functions: 
1. My [Udacity/Facebook Secure Private AI](https://www.udacity.com/facebook-AI-scholarship) Keystone Project
2. My [SIIM-ACR Pneumothorax Segmentation Kaggle contest](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) submission
# Steps Completed
Each cell represents a step
1.	Importing libraries, included PyTorch and PySyft
2.	Printing what I have in my input folder 
3.	Calculating and printing the number of test and training files
4.	Displaying 10 images
5.	Determining how many images were labeled out of the total number of images
6.	Calculating how many images were missing
7.	Determining the mean of the test images
8.	Determining the mean of the training images
9.	Reading in `.csv` data and clean the column names
10.	Defining a function that converts a diacom file to a Python `dict` by:
    1.	Creating an empty dict called `data`
    2.	Parsing the fields with meaningful information 
    3.	Looking for annotation if this feature is enabled 
11.	Applying this function to the train data and converting the result to a pandas DataFrame
12.	Applying this function to the test data and converting the result to a pandas DataFrame
13.	Printing the number of x-rays with missing labels 
14.	Defining a function that cleans the pandas DataFrame by:
    1.	Printing the tail 
    2.	If the mode is set to train, extracting the labels, dropping 4 columns, and one-hot encoding the `encoded_pixels_list` column
    3.	If the mode is not set to train, setting the labels variables equal to 0 and drop 3 columns 
    4.	Replacing the `patient_sex` column with 0 if the original value was M and 1 if the original value was F
    5.	One-hot encoding the `id` column from a list of integers that are separated by periods   
    6.	One-hot encoding `pixel_spacing` column
    7.	Printing the tail
15.	Applying this function to the train data 
16.	Applying this function to the test data 
17.	Creating the class Arguments, which is a bunch of settings that I will later use for my NN, and setting the variable `args` equal to an instance of this class
18.	Setting up a `hook`, `client`, 2 virtual workers, and `cyrpto_provider` using `syft`
19.	Creating a base by converting my training data into a numpy array and then a torch dataset; federating this base and converting this federated base into a `FederatedDataLoader`
20.	Creating another torch dataset by converting the test dataframe to a numpy array
21.	Defining the class `Net`, which has 2 parts: 
    1. `__init__` - this part defines all the properties (the convolutional, pooling, and fully connected layers) of the class
    2.	`forward` - this part is the heart of the NN. This part takes the data as `x` and: 
        1.	Converts `x` to a float
        2.	Runs `x` through the first convolutional layer, the relu function, and then pooling layer
        3.	Runs `x` through the second convolutional layer, the relu function, and then pooling layer
        4.	Runs `x` through the third convolutional layer, the relu function, and then pooling layer
        5.	Converts the `shape` of `x` from `[Y, X]` to `[-1,  64*6*1]`
        6.	Runs `x` through the first fully connected layer and the relu function
        7.	Runs `x` through the fully convolutional layer and the relu function
        8.	Runs `x` through the fully convolutional layer and the relu function
22.	Defining a `train`ing function that:
    1.	Sets the model in the correct mode
    2.	For each batch index, data, and target in `federated_train_loader`:
        1.	Sends the model to the data’s location
        2.	Sends the data and target to `device`
        3.	Clears the `grad` from the optimizer 
        4.	Unsqueezes the data *twice* (adds another dimension *twice*); once is in-place while the other is being fed directly into the model 
        5.	Sets `loss` to be equal to the null loss function
        6.	Moves `loss` backward and steps the optimizer 
        7.	Gets the model back
        8.	If the reminder (modulo) of the current batch index and the log interval property we set from `Args` is zero:
               1.	Get loss back
               2.	Print training statistics 
23.	Defining a `test`ing function that:
    1.	Sets the model in the correct mode 
    2.	Declares test los and number correct to be zero
    3.	With the gradients cleared and for data as well as target in the test data set:
        1.	Sends the target and data to the correct location (`device`)
        2.	Feeds the data into the model 
        3.	Sets `test_loss` equal to the null loss function’s sum
        4.	Sets `pred` equal to the model’s output
        5.	Sets `correct` equal to itself plus the sum of the prediction (`pred`)
    4.	Sets `test_loss` equal to itself divided by the `len`gth of the test dataset
    5.	Prints summary statistics 
24.	Moves an instance of our `Net` class `to` the device, defines the optimizer to be SGD, prints the model, and for each `epoch`:
    1.	Applies the training function 
    2.	Applies the testing function 
