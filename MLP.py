import numpy as np
from numpy.random import randn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

from helper import *



class MLP:
    
    def __init__(self, x_dim, units_per_layer):
        
        # dimension of a single input x 
        self.x_dim = x_dim  
        self.units_per_layer = units_per_layer

        # list to store the layers
        self.layers = []

        # rand initilise parameters for each layer
        for k, h_n in enumerate(units_per_layer):

            layer = {}  # each layer is a dict containing its params

            # retrieve dimension of previous actiation layer
            h_n_prev = x_dim if k == 0 else units_per_layer[k - 1]

            layer["W"] = randn(h_n, h_n_prev) * 0.01
            layer["b"] = randn(h_n, 1) * 0.01  # we could have also doen 0s

            self.layers.append(layer)  # store the layer dict in the layers list
        

    
    def forward(self, data):
        
        for l_idx, this_layer in enumerate(self.layers):

            # retrieve previous activation layer's activation
            h_prev = data if l_idx == 0 else h

            # compute the linear part
            z = np.dot(this_layer["W"], h_prev) + this_layer["b"]

            # compute the activation for the current layer
            if l_idx == len(self.layers)-1:
                h = softmax(z)
                
            else:
                h = sigmoid(z)

            # store the activation and linear part in the dict l for layer i
            # we will use this in backprop
            this_layer["h"], this_layer["z"] = h, z

        return h
    
                    
    def backprop(self, x_train, y_train, h):

        try:
            m = y_train.shape[1]
        except:
            m = 1

        # loop through the layers in reverse order
        for l_idx, this_layer in enumerate(reversed(self.layers)):

            # retrieve the previous layer's activation as computed in forward
            try:
                h_prev = self.layers[-2 - l_idx]["h"]
            except:
                h_prev = x_train

            # compute the gradient of the softmax layer
            if l_idx == 0:
                dz = h - y_train
            
            # compute the gradient of the sigmoid layers
            else:
                dz = dh * sigmoid(this_layer["z"]) * (1 - sigmoid(this_layer["z"]))


            # compute the gradient of dL/dW and store in the layer's prams dict
            this_layer["dW"] = np.dot(dz, np.transpose(h_prev)) / m

            # compute the gradient of dL/db and store in the layer's prams dict
            this_layer["db"] = np.sum(dz, axis=1, keepdims=True) / m

            # compute dL/dh__{i-1} and update dh for the next layer backprop
            # calculation
            dh = np.dot(np.transpose(this_layer["W"]), dz)


    def update_parameteres(self, lr=0.001):

        for this_layer in self.layers:
            this_layer["W"] = this_layer["W"] - lr * this_layer["dW"]
            this_layer["b"] = this_layer["b"] - lr * this_layer["db"]

    
    
    def zero_grads(self):
        """
        Set the gradients arrays for finite difference to zero
        """

        for this_layer in self.layers:
            this_layer["dW"] = np.zeros_like(this_layer["W"])
            this_layer["db"] = np.zeros_like(this_layer["b"])
            

    def batches(self, x_train, y_train, batch_size):
        """
        Generator to produce batches for mini-batch GD
        Randomly shuffles the whole data set first
        """

        # shuffle
        idx = np.random.permutation(x_train.shape[1])
        x_rand = x_train[:,idx]
        y_rand = y_train[:,idx]

        # compute the total number of batches
        n_batches = int(np.ceil(x_train.shape[1] / batch_size))


        for b in range(n_batches):
            
            #find the start and end of this batch
            start = batch_size * b
            end = min((batch_size) * (b+1), y_train.shape[1])
            
            x = x_rand[:, start: end]
            y = y_rand[:, start: end]
           
            yield x, y
            
            
    def train(self, x_train, y_train, x_dev, y_dev, batch_size=None, verbose=False, epochs=100, lr=.1):
        """
        train the model

        Args:
            x_train: training samples
            y_train: training labels
            x_dev: development samples for evaluation
            y_dev: development labels
            batch_size: mini batch size, full batch if None
            epochs: number of epochs to run traing for
            lr: learning rate
        """
        
        train_hist = []
        dev_hist = []

        if batch_size is None: batch_size = y_train.shape[1]

        # init Dataframe to store training results
        self.results = pd.DataFrame(dtype=float, columns=["train_loss", "dev_loss", "train_acc", "dev_acc", "lr"])

        i = 0  # count iterations, i.e. every batch

        # iterate through the number of epochs
        for e in range(epochs):

            # iterate through the batches in this epoch
            for x_batch, y_batch in self.batches(x_train, y_train, batch_size):

                m = y_batch.shape[1]
                # forward pass
                out = self.forward(x_batch)

                # compute loss
                loss = self.cross_entropy(out, y_batch)

                # perform backprop to compute grads
                self.backprop(x_batch, y_batch, out)

                # perform updates for W and b of every layer
                self.update_parameteres(lr=lr)

                # compute all metrics on train and dev sets
                acc = accuracy(self.predict(x_train), y_train)
                dev_acc = accuracy(self.predict(x_dev), y_dev)
                dev_out = self.forward(x_dev)
                dev_loss = self.cross_entropy(dev_out, y_dev)
                train_out = self.forward(x_train)
                train_loss = self.cross_entropy(train_out, y_train)

                # store all metrics in the results Dataframe
                self.results.loc[i, "train_loss"] = train_loss
                self.results.loc[i, "dev_loss"] = dev_loss
                self.results.loc[i, "train_acc"] = acc
                self.results.loc[i, "dev_acc"] = dev_acc
                self.results.loc[i, "lr"] = lr
                self.results.loc[i, "m"] = m
                i += 1 #update iter counter

                # print metrics for current iteration if verbose
                if verbose and i % verbose == 0:
                    print(f"Epoch {e}")
                    print(f"=====Iter {i}======================")
                    print(f"Train Loss: {loss}")
                    print(f"Dev Loss: {dev_loss}")
                    print("Train accuracy: ", acc)
                    print("Dev accuracy: ", dev_acc)
                    print("Learning rate: ", lr)
                    
                    
            # check convergence
            train_acc = accuracy(self.predict(x_train), y_train)
            if len(train_hist)>300 and train_acc < train_hist[-4] + 0.001:
                break
            else:
                train_hist.append(train_acc)
                
            # check overfitting
            dev_acc = accuracy(self.predict(x_dev), y_dev)
            if len(dev_hist)>300 and dev_acc < dev_hist[-4] + 0.001:
                break
            else:
                dev_hist.append(dev_acc)
                    
    def predict(self, x):
        """
        Make predictions

        -- Input --
        x: x input to predict labels
        
        -- Returns --
        y_hat: predicted labels : 0 or 1
        """
        out = self.forward(x)
        y_hat = np.argmax(out, axis=0)
        return y_hat   

    
    def plot_results(self, rolling=None):
        """
        Plot all metrics stored during process
        Print metrics for the iteration with the best dev accuracy

        -- Input --
        rolling: window size to plot with moving average.
        """

        r = self.results.copy()

        max_dev_acc = max(r["dev_acc"])
        to_print = r[r["dev_acc"] == max(r["dev_acc"])]

        print("Results for the first iteration achieving Best Dev Accuracy")
        print(to_print.iloc[0])

        plot_results = self.results.copy()

        if rolling:
            for m in ["train_loss", "train_acc", "dev_loss", "dev_acc"]:
                plot_results[m] = plot_results[m].rolling(window=rolling).mean()

        fig, ax = plt.subplots(1, 2, figsize=(14,6))
        sns.lineplot(data=plot_results[["train_loss", "dev_loss"]], ax=ax[0])
        sns.lineplot(data=plot_results[["train_acc", "dev_acc"]], ax=ax[1])
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('Loss')
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Accuracy')
        
        
    def cross_entropy(self, p, y):
        """
        Compute the categorical cross entropy loss function

        -- Input --
        p: out put of the forward pass of our model
        y: true labels

        -- Return --
        loss: the negative loglikelihood
        """

        p += 1e-8
        
        pred = softmax(p)
        
        return np.mean(np.sum(-np.log(pred)*y, axis=0))
    
    def calc_errors(self, x_train, y_train, x_dev, y_dev):
        # compute all metrics on train and dev sets
        train_err = 1 - accuracy(self.predict(x_train), y_train)
        test_err = 1 - accuracy(self.predict(x_dev), y_dev)

        return train_err, test_err
        
def one_hot(y, nclasses=10):
    y = y.astype(int)
    oh = np.zeros((y.size, nclasses), dtype=int)
    oh[np.arange(y.size), y] = 1

    return oh.T
        
def softmax(x):
    """Return the softmax function of x"""
    h = np.exp(x)
    h /= np.sum(h, axis=0, keepdims=True)
    return h
        
def accuracy(y_hat, y):
    """
    Compute the accuracy
    
    -- Inputs --
    y_hat: predicted labels
    y: true labels
    """

    return np.sum(y_hat == np.argmax(y, axis=0)) / y.shape[1]