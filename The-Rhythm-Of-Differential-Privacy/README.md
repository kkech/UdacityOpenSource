# The-Rhythm-Of-Differential-Privacy
------------------------------------------------------------------------------

![High-level Architecture](https://raw.githubusercontent.com/OpenMined/PySyft/dev/art/PySyft-Arch.png)

## Summary
------------------------------------------------------------------------------
The goal of this project is to utilize the PySyft framework to apply differential privacy, on both a local and global scale, and compare the accuracy between models trained with and without these processes.  We have also examined applying Laplacian noise by utilizing federated learning on a pre-existing Android Worker and graphing the ouputs of the models.

We began with the tutorial [PySyft for Android](https://towardsdatascience.com/pysyft-android-b28da47a767e) by Jose Corbacho and built implemented additional functionality with it.  GraphView was implemented on the Android Worker in order to analyze the graphs of the noise.

## Continuing Work:
------------------------------------------------------------------------------
There are several aspects of this project that we wish to implement in the future including:
* The comparison of Gaussian and Laplacian noise application in terms of accuracy and rate of training
* Encrypted Aggregation
* More complete implementation of a CNN with noise in a federated learning situation.

## Reference
------------------------------------------------------------------------------

1. [**The Algorithmic Foundations of Differential Privacy**](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
1. [**PySyft for Android**](https://towardsdatascience.com/pysyft-android-b28da47a767e)   
2. [**These represent working upon grid nodes and android apps**](https://github.com/OpenMined/Grid/tree/dev/examples/experimental) 
3. [**General tutorials on using pysyft and performing various operations**](https://github.com/OpenMined/PySyft/tree/dev/examples/tutorials)
4. [**Android Worker**](https://github.com/OpenMined/AndroidWorker)
5. [**PySyft**](https://github.com/OpenMined/PySyft)  
7. [**All serialization**](https://github.com/OpenMined/PySyft/blob/887e558fe094f7245421a23c9da65370fde2f121/syft/serde/serde.py) |   [**Serializations for torch tensors**](https://github.com/OpenMined/PySyft/blob/887e558fe094f7245421a23c9da65370fde2f121/syft/serde/torch_serde.py)


## Contributors   
------------------------------------------------------------------------------

| Name | Slack Name |
| ------------------------- | ------------------------- |
| [Sarah Majors](https://github.com/sfmajors373) | Sarah Majors | 
| [Harkirat Singh](https://github.com/Harkirat155) | Harkirat |
| [Hsin Wen Chang](https://github.com/Polarbeargo) | Bearbear |
| [Halwai Aftab Hasan](https://github.com/ahkhalwai) | Halwai Aftab Hasan |

