# Differentially Private Federated Learning A client Level Perspective.

This Notebook Implements the paper: Differentially Private Federated Learning: A Client Level Perspective (https://arxiv.org/abs/1712.07557)

Abadi et al propose a Differential Private SGD to train Deep Learning models in a federated way. The idea was to apply some noise at each gradient to make it (epsilon, delta) differentiable private. Geyer et al takes this idea one step ahead and works on an algorithm that is not (epsilon, delta) differentiable private at gradient level but at the client level. This makes that the model can converge faster, with less noise apply to the optimization steps and continues to protect the privacy of the different clients that are participating in the training. 