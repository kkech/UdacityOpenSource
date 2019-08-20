# SMS Spam Detection

Author: [Govind Dixit](https://github.com/GOVINDDIXIT)

*Federated Learning is a machine learning setting where the goal is to train a high-quality centralized model with training data distributed over a large number of clients each with unreliable and relatively slow network connections.*

*In this project I have used Federated Learning with the PyTorch extension of PySyft for a classification task with a simple 1-layer GRU.* 

**Dataset:**
[Link](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

**Dataset Description:**
The data used for this project was the SMS Spam Collection Data Set available on the UCI Machine Learning Repository. The dataset consists of 5500 SMS messages, of which around 13% are spam messages.

*The objective here is to simulate two remote machines (that we will call Bob and Anne), where each machine have a similar number of labeled data points (SMS labeled as spam or not).*


### Conclusion / Result

*We can see that with the PySyft library and its PyTorch extension, we can perform operations with tensor pointers such as we can do with PyTorch API (but for some limitations that are still to be addressed).*

*Thanks to this, we were able to train spam detector model without having any access to the remote and private data: for each batch we sent the model to the current remote worker and got it back to the local machine before sending it to the worker of the next batch.*

*We can also notice that this federated training did not harm the performance of the model as both losses reduced at each epoch as expected and the final AUC score on the test data was above 97.5%.*

*There is however one limitation of this method: by getting the model back we can still have access to some private information. Let's say Bob had only one SMS on his machine. When we get the model back, we can just check which embeddings of the model changed and we will know which were the tokens (words) of the SMS.*

*In order to address this issue, there are two solutions: Differential Privacy and Secured Multi-Party Computation (SMPC). Differential Privacy would be used to make sure the model does not give access to some private information. SMPC, which is one kind of Encrypted Computation, in return allows you to send the model privately so that the remote workers which have the data cannot see the weights you are using.*

Thank You 🤞
