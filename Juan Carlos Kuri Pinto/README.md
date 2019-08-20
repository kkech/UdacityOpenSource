# Diagnosing Acute Inflammations of Bladder
**SOURCE: https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations**

**Abstract:** This machine learning system can diagnose 2 acute inflammations of bladder. The medical dataset contains features and diagnoses of 2 diseases of the urinary system: Inflammation of urinary bladder and Nephritis of renal pelvis origin. This medical dataset truly needs privacy! Because we cannot divulge the sexually-transmitted diseases of patients. So, all we learned about PySyft and OpenMined will be applied in this project.

**Software Requirements:**
- Python 3.7.3 (imports: urllib.request, numpy, torch, torch.autograd, torch.nn, torch.nn.functional, torch.optim, matplotlib.pyplot, syft)
- PyTorch 1.1.0
- PySyft 

**Instructions:** Download and run the Jupyter notebook **Bladder Dataset.ipynb**
<p align="center">
 <img src="images/bladder.jpg" title="Bladder">
</p>

## FEDERATED LEARNING WITH A TRUSTED AGGREGATOR

Let's assume we have 4 hospitals. (The datasets will be split in 4, randomly.) And the 4 hospitals cannot share their cases because they are competitors. Hence, the ML model will be learned in a federated way by sending the model updates to a trusted aggregator that will average the model updates. The updated model will be sent back to each hospital in order to train the ML model in an iterative way. Only the ML model will be shared whereas the cases of each hospital will be kept private and will train model updates in a local way.<br>
<br>

<center>
 <img src="images/federated-learning.png">
 Federated Learning - Image taken from <a href="https://www.intel.ai/federated-learning-for-medical-imaging/">https://www.intel.ai/federated-learning-for-medical-imaging/</a>
</center>
