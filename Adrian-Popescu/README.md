# Encrypted classification (with Bob and Alice) project
This repository contains the Jupyter notebook for my Udacity Facebook Secure and Private AI Challenge project.

Description:
1. A simple fully connected classifier () is built using PyTorch and trained on the MNIST dataset to classify handwritten digits.
 
2. The trained model is then encrypted using private additive sharing and send to two workers (Bob and Alice) created using PySyft.
 
3. A testing dataset is also encrypted and sent to Bob and Alice for inference.
 
4. The inference is privately executed on the two workers, and only the accuracy of results is decrypted and printed.

5. The project shows that it can infer with good accuracy (~98%) without leaking any information about the model or the testing data.

## Dependencies

The project requires the PySyft library in order to run. The prerequisites are easier installed using conda:

`conda create -n pysyft python=3`

`conda activate pysyft`

`conda install jupyter notebook`


### pip

To install syft with pip, you can issue `pip install syft`.

### Contributors

| Name | Slack |
| ------ | ------ |
| Adrian Popescu | Adrian Popescu |