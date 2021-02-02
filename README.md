December 2020

# Kernel perceptron
A kernel perceptron implementation with polynomial and Gaussian kernels. For compuational efficiency, the kernels are computed a priori and looked-up by advanced indexing. Parameter optimization for the polynomial degree and Gaussian kernel-width are done through 5-fold cross validation. To gauge the kernel perceptrons performance, k-Nearest Neighbour and Multi-Layer Perceptron algorithms are implemented. All algorithms are implemented using only the `Numpy` library (e.g. no PyTorch/TensorFlow, scikit-learn, etc.).
