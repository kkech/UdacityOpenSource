# Syft Web Worker

This project is created for Udacity's [Secure and Private AI](https://www.udacity.com/course/secure-and-private-ai--ud185) challenge course.
It explores possibility to have PySyft worker running in the web browser, 
which potentially expands PySyft functionality to a very large user-base. 
The back-end for tensor operations in browser is tf.js.

The current project's state is 'proof of concept' where tensors can be moved to/from browser and a limited set of tensor operations is possible. 
Further direction is to try to use web client for federated learning.

## Demo
Follow this [Colab jupyter notebook](https://colab.research.google.com/github/vvmnnnkv/syft-js-worker/blob/master/Syft%20Web%20Client%20Demo.ipynb).

For local installation:
```
pip install -Ur requirements.txt
python run_socketio_server.py
```
Update `index.html` and notebook code with server's URL (e.g. `http://localhost:5000`).

## Work in progress
There's zero infrastructure work done (e.g. dependency management, building, etc.), and javascript is tested in Chrome only.

## Acknowledgments
 * [syft.js](https://github.com/OpenMined/syft.js) - inspiration :)
 * [PySyft Android Worker](https://github.com/OpenMined/AndroidWorker) - server script and some PySyft protocol details
 * Numpy to array code borrowed from: https://gist.github.com/nvictus/88b3b5bfe587d32ac1ab519fd0009607
 * Array to numpy code borrowed from: https://github.com/propelml/tfjs-npy/blob/master/npy.ts
 * Thanks Udacity's [Secure and Private AI](https://www.udacity.com/course/secure-and-private-ai--ud185) challenge course and Andrew Trask for great content

