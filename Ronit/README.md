
# Project - Encrypted and Automated Essay Grading
## Description
- Grader, a website which automatically grades an essay written by a student using a Deep Learning model, namely an Long-Short-Term-Memory network. 
- Both the model and the data are encrypted using Secure Multi Party Encryption (SMPC).
- The dataset is from Kaggle ASAP competition which was provided by The Hewlett Foundation.
- The training was performed on the above dataset using SMPC with PySyft's `share` function.
- The prediction is then done on the client side by importing the encrypted model and using it to predict the grade for the input essay (after encrypting the essay).
---

## Website ScreenShots

<img src="https://github.com/mankadronit/UdacityOpenSource/blob/Ronit/Ronit/assets/SC1.png" width="100%">



<img src="https://github.com/mankadronit/UdacityOpenSource/blob/Ronit/Ronit/assets/SC2.png" width="100%">



<img src="https://github.com/mankadronit/UdacityOpenSource/blob/Ronit/Ronit/assets/SC3.png" width="100%">


## Model Description
The model used is an LSTM network followed by a a fully connected Dense layer with 1 output nodes. 

<img src="https://github.com/mankadronit/UdacityOpenSource/blob/Ronit/Ronit/assets/lstm.png" width="100%">


## Key Takeaways
- Secure Multi Party Computation is a game changer in the Machine Learning as a service field (MaaS).
- SMPC paired with Differential Privacy can revolutionalize services like medicine prescription.

## Sample PyTorch Code

### Step 1: Create the workers and crypto providers
``` 
import torch
import syft as sy  # import the Pysyft library
hook = sy.TorchHook(torch)  # hook PyTorch to add extra functionalities like Federated and Encrypted Learning

# simulation functions
def connect_to_workers(n_workers):
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]
def connect_to_crypto_provider():
    return sy.VirtualWorker(hook, id="crypto_provider")

workers = connect_to_workers(n_workers=2)
crypto_provider = connect_to_crypto_provider()
```

### Step 2: Create the private dataloaders
```
def secret_share(tensor):
  """
  Transform to fixed precision and secret share a tensor
  """
  return (
      tensor
      .fix_precision(precision_fractional=args.precision_fractional)
      .share(*workers, crypto_provider=crypto_provider, requires_grad=True)
  )
    

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True)

# Convert to integers and privately share the dataset
private_train_loader = []
for data, target in train_loader:
    private_train_loader.append((
        secret_share(data), secret_share(target)
    ))
```

### Step 3: Create the model and encrypt it using the `share()` function.
```
model = nn.Sequential(nn.Embedding(len_voc, embed_dim),
                       nn.LSTM(64, 2, bidirectional=True),
                       nn.Linear(128 ,1),
                       nn.ReLU())
model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)

```

## Tools Used
> ⚠️ **Note**: The model in this repo was 
coded with Keras for demonstration purposes. A production version will be made using Pytorch and PySyft. This repo doesn't contain the code for the encryption process.

- Keras v2.2.2
- Django v2.1
- Gensim v3.5 
- Pandas
- Numpy
- NLTK

```
pip install keras
pip install django
pip install gensim
pip install pandas
pip install numpy 
pip install nltk
```

### Performance
The accuracy is calculated by **Quadratic Weighted Kappa(QWK)**, which measures the agreement between two raters. The model architecture consists of 2 Long Short Term Memory(LSTM) layers with a Dense output layer. The final layer uses the Relu activation function. The QWK is calculated by training model on the dataset using 5-Fold Cross Validation and taking the average for all five folds.


## Installation 
- Clone the repo
- Just run the **Jupyter Notebook** to train the model.
- To run the Django App cd ./ into the **mysite** folder and run ``` python manage.py runserver```

## References
1. [A Neural Approach to Automated Essay Scoring](http://aclweb.org/anthology/D/D16/D16-1193.pdf) </br>
2. [Automatic Text Scoring Using Neural Networks](https://arxiv.org/pdf/1606.04289.pdf)
3. Secure and Private AI course on Udacity.