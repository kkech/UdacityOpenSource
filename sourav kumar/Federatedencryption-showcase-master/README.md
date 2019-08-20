# Project Objective :   
## This project aims to achieve [federated learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) training of model and then testing on [encrypted dataset and model](https://iamtrask.github.io/2017/03/17/safe-ai/) using [pytorch](https://pytorch.org/) and [pysyft](https://github.com/OpenMined/PySyft).

## Dataset : MNIST 
> (Intent behind using the simple dataset is the main motive which is to combine privacy preserving techniques using pysyft)

![](pics/google.gif)   

## Basic overview of federated learning :   

Have you ever wondered how Google keyboard predicts the probability of the words you will type after you just start typing ?
Or how the Apple iOS detects the most correct emoji you want to use in a conversation.

![](pics/ios.png)   

There are similar other things where we can see the use of federated learning like :
* wearable medical devices
* predictive maintenance (automobiles / industrial engines)
* ad blockers / autotomplete in browsers (Firefox/Brave)
* app company (Texting prediction app)
* so what exactly is federated learning ?

**Federated Learning is a collaborative form of machine learning where the training process is distributed among many users and not based totally on a single server. Here , the server is the main control for coordinating everything but most of the work for training process is done by federation of users.**  

Normally , one would send all the data to a central server for training the model created on the server and then the model is trained but for some security reasons we might not want to send all of the data in the hands of some other organisations like all the things we type in daily on the phone.
So, federated learning is one of the solution , though not the only one.
In the context of federated learning process, we call the devices or a remote machine ‘workers’.
You might understand in terms like this say , you have a boss and that boss in your organisation has the main role of supervising things there but since he is the busiest due to other stuff like meetings and all , he keeps some persons who are employees working there.
These employees work for different dept. and also share some of the data with the Boss and organisation and therefore making contribution towards overall development of the company / organisation.   

![](pics/fed1.png)    


Let’s now learn how the process actually works :
* First, the model initializes the weights on the server , by using any method like random, He, Xavier initialization.
* Then a random sampling is done to select the users to improve the model by training the model on the local datasets available on the local devices / remote machines.
* It means that the ‘workers’ can be called as real producer of the dataset on which the model actually learns.
* Then each sampled users receives the model and starts training on the local dataset and computes the model updates like new weights and gradients which are model parameters.
* All of these computations of model updates are sent back to the server and then take weighted average with respect to no of training examples that clients (devices) used.
* Then the server applies the updated model to the old model by using optimization algorithms like gradient descent or any other form.

_Now, what are the hyperparameters we are concerned with ?
The model architectures used for other learning process will remain almost same although it’s a hit and trial process but there is yet another hyperparameter called ‘communication rounds’.
Since this sending of model and getting received it back with updates from billion of users at a time is a communication channel rounds and hence , how many users are being sampled in each round influences how many rounds of communication is required until the convergence is reached._   

![](pics/fed2.png)      

Let’s talk about challenges faced during federated learning :
* Data is distributed across millions of devices in a highly uneven fashion
* Different users could be creating different non identical distributions
* These devices have significantly higher-latency, lower-throughput connections and are only intermittently available for training.
_Now you will be wondering that while training takes place my phone performance would be impacted.
But surprisingly , nothing impacts the performance.
To understand why read this extract which is taken from Google AI blog:
“Careful scheduling ensures training happens only when the device is idle, plugged in, and on a free wireless connection, so there is no impact on the phone’s performance”_    

Now , let’s see some of the changes introduced in normal federated learning to improve the privacy and move it to next level of enhancement :
* Users are sampled randomly and not always equal , this ensures there is no bias.
* Noise is added to the final output update of the model by applying globally differentially private algorithms.
* There is new protocol for transfering data Secure Aggregation protocol which uses cryptographic techniques when decrypting the updates on the server and the server only does it when if 100s or 1000s of users have participated. No individual phone’s update can be inspected before averaging.

## Basic overview of encrypted deep learning :   

![](pics/fed3.png)   

Okay before we move on , let’s set up our flow in which we will be going —
* Brief introduction to modular arithmetic
* Secret sharing technique
* Encrypted deep learning with pysyft
**Modular arithmetic** — why is it important here ?
Let me ask you a simple question , suppose that a clock is showing time 2'O clock , and i ask what will be time after 15 hours , you might tell the correct answer as 5'O clock.     

![](pics/fed4.jpeg)    

But ask a person who don’t know how to read a clock and he might answer 17'O clock , but hey 😕 we don’t have 17 in the clock !   
So , have you observed that when you do add 15 + 2 you get 17 but the thing is that clocks work on mod 12 arithmetic also represented as “%”.   

Remember classical arithmetic teaches us that if we add any number “a + b”
Then we would obtain a result which will be greater than the numbers themselves but it is not the case with modular arithmetic.
Let’s see this in the context of above example , when we take 17 % 12 or 17 mod 12 then we will obtain 5 which is also in the range under 12.   

> So , one of the key takeaways is that whenever we take mod of some number like mod m it is guaranteed that our result will not overflow this chosen number m and all the overflowed numbers will wrap around this range.  

Modular arithmetic helps in many recent technological advancements like Blockchain technology , Cryptographic techniques , Privacy preserving Deep learning , Machine learning etc.   

**Secret sharing technique** :    
Let’s see this in more pythonic way (you are going to learn fascinating things about how modular arithmetic helps in encrypting numbers allowing us to perform computations over them secretly 🤐):   

![](pics/fed5.jpeg)     

We see that actually we add 5 and 3 instead of 15 and 13 , thus encrypting the actual numbers and similarly for subtraction.
So , let’s say we want to perform computations over some numbers but keeping them secret.
So , we take a number say 5 and then split this number using a simple formula shown below :   

![](pics/fed6.jpeg)    

> Field is basically limit size or range within with we wanna wrap our numbers to stop overflowing. 
Generally , we prefer it to be very large prime number.   

ow we have encrypted “x” here.
But wait , how to decrypt it ?
There’s a simple elegant way of doing this in modular arithmetic -
we sum up all the shares and take mod field to get our number back.  

![](pics/fed7.jpeg)      

**Encrypted deep learning with Pysyft** :   

So, the thing is that there are various security and privacy issues when we train the data , what if the server on which we upload our data to train sees our model parameters and use them for training their own model to outrun us in say any online competitions , or there are organisations which are hesitating to share their model to someone else just for inference but can do so if their privacy is totally ensured.  

Thus , we need a way to encrypt both our data and model so that organisations can use other’s models to get their inference (predictions) because not everyone in the industry likes to create models 🙂   

So, here comes a handy library “Pysyft” which helps to secure our data and models both , allowing us to perform encrypted computations like training and inference without ever disclosing anything about data and model using all the arithmetic and secret sharing that we earlier discussed under the hood.     

Also , we create a few workers (named client, bob, and alice). 
We define the crypto_provider who gives all the crypto primitives we may need.   

The secret sharing that we discussed earlier is being used here to privately share the dataset across the workers we have created using pysyft’s inbuilt function **“fix_prec()”**.   

Excerpt taken from openmined blog:   

> “ SMPC uses crypto protocols which require to work on integers. We leverage here the PySyft tensor abstraction to convert PyTorch Float tensors into Fixed Precision Tensors using .fix_prec(). For example 0.123 with precision 2 does a rounding at the 2nd decimal digit so the number stored is the integer 12.”   


(I have used colab for running all of the above code:)  

## Results snap : 

![](pics/result.gif)      





