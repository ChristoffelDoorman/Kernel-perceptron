import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'20',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


from kNN import kNN, kNN
from helper import *


# load in the data
data = pd.read_table("data/zipcombo.dat", sep="\s+")
data = np.array(data)

def task1(param_set, runs=20):
    """
    This function produces basic results for the k-NN algorithm
    
    Input:
    param_set: arr or list -- the parameters of the k-Nearest Neighbour algorithm
    runs: int -- number of runs to average 'best' k over
    
    Returns 4 lists containing the train errors, std and test errors, std of all parameters
    """
    
    # save the means and stds of all parameters
    train_means = []
    train_stds = []
    test_means = []
    test_stds = []
    
    # loop through the parameters, running multiple iterations per param
    for k in tqdm(param_set, 'k'):
        
        # save errors of this parameter
        all_train_errors = np.zeros(runs)
        all_test_errors = np.zeros(runs)
        
        # run multiple iterations for averaging
        for this_run in tqdm(range(runs), 'run'):
            
            # split the dataset into train and test
            train_set, test_set = train_test_split(data, test_size=0.2)
            train_set = LabelledDataset(train_set)
            test_set = LabelledDataset(test_set)
            
            # initialize kNN
            knn = kNN(train_set, test_set, k)
            
            # calculate and store errors
            train_error = knn.accuracy(train_set.data, train_set.labels)
            test_error = knn.accuracy(test_set.data, test_set.labels)
            all_train_errors[this_run] = train_error
            all_test_errors[this_run] = test_error
            
        # calculate means and standard devs of errors
        train_error_mean = np.mean(all_train_errors)
        train_error_std = np.std(all_train_errors)
        test_error_mean = np.mean(all_test_errors)
        test_error_std = np.std(all_test_errors)
        
        print('d =', k)
        print('train error: %f ± %f' %(train_error_mean, train_error_std))
        print('test  error: %f ± %f\n' %(test_error_mean, test_error_std))
        
        train_means.append(train_error_mean)
        train_stds.append(train_error_std)
        test_means.append(test_error_mean)
        test_stds.append(test_error_std)
        
    return train_means, train_stds, test_means, test_stds


def task2(param_set, runs=20, kval=5):
    """
    This function executes a kval-Fold cross-validation to tune
    the parameter 'k' of a k-Nearest Neighbour. The cross-validation 
    is directly implementated withouth the use of libraries. 
    
    Input:
    param_set: arr or list -- the parameters to cross validate over
    runs: int -- number of runs to average 'best' k over
    kval: int -- number of folds in cross-validation
    
    Returns:
    best_k: 1d-array -- array containing runs-number of 'best' k
    test_errors: 1d-array -- array containing test errors for every k
    
    """
    
    # save best d and corresponding test error of every run
    best_k = np.zeros(runs)
    test_errors = np.zeros(runs)

    for this_run in tqdm(range(runs), 'Run'):
        
        error_best_k = float('inf')
        this_best_k = 0
        
        # randomly split the dataset into .8 train and .2 test
        train_set, test_set = train_test_split(data, test_size=0.2)

        # split the random train set into kval-fold validation sets
        cross_val_sets = np.array_split(train_set, kval, axis=0)
        
        # convert train and test set to input format of Kernel_perceptron
        train_set = LabelledDataset(train_set)
        test_set = LabelledDataset(test_set)
    
        for k in tqdm(param_set, 'k'):
            
            # sum the k-fold validation errors
            total_val_error = 0
            valset_count = 0
            
            for val_idx in range(kval):
                thisfold_train_set = cross_val_sets.copy()
                
                # take the next fold as validation set
                val_set = thisfold_train_set.pop(val_idx)
                
                # merge the other folds as training set
                thisfold_train_set = np.concatenate(thisfold_train_set)

                # convert data sets to input format of kNN class
                thisfold_train_set = LabelledDataset(thisfold_train_set)
                val_set = LabelledDataset(val_set)
                
                # initialize kNN
                thisfold_kNN = kNN(thisfold_train_set, val_set, k)

            
                # train kernel perceptron and find train and test errors
                thisfold_train_error = thisfold_kNN.accuracy(thisfold_train_set.data, thisfold_train_set.labels)
                thisfold_val_error = thisfold_kNN.accuracy(val_set.data, val_set.labels)

                # add validation error to total
                total_val_error += (thisfold_val_error * val_set.size)
                valset_count += val_set.size
            
            # calculate weighted validation error from k-folds
            val_error = total_val_error / valset_count
            
            # check if this d is better
            if val_error < error_best_k:
                this_best_k = k
                error_best_k = val_error
                    
        # save best parameter d of this run
        best_k[this_run] = this_best_k       
        
        # train on full train set with best d
        kNN = kNN(train_set, test_set, this_best_k)
        train_error = kNN.accuracy(train_set.data, train_set.labels)
        test_errors[this_run] = kNN.accuracy(test_set.data, test_set.labels)

    
    print('Run', this_run+1)
    print('d^* mean: %f ± %f' %(np.mean(best_k), np.std(best_k)))
    print('test_err: %f ± %f' %(np.mean(test_errors), np.std(test_errors)))

    
    return best_k, test_errors