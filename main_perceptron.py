import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

from helper import *
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


from kernel_perceptron_Copy1 import Kernel_perceptron


# load in data
data = pd.read_table("data/zipcombo.dat", sep="\s+")
data = np.array(data)


def task1_perceptron(data, runs, kernel_func='polynomial', classification_method='OvA', param_set=range(1,8)):
    """
    Function to execute questions 1, 5 and 6.
    Train and test a polynomial or Gaussian kernel perceptron. For computational
    efficiency, all kernel matrices are computed at the beginning of the program,
    and indices are used to lookup the kernel values.
    
    -- Input --
    data: [datasize, datadim] array -- all data
    runs: int -- number of runs to average over
    kernel_func: str -- 'polynomial' or 'Gaussian'
    classification_method: str -- 'OvA' (one vs all) or 'OvO' (one vs one)
    param_set: list or arr -- set containing all kernel parameters
    
    -- Return --
    error_dict: dict -- dictionary containing { parameter: 
                                                { train_error: , train_std:, test_error: , test_std: } }
    """
    
    # keep track of indices for kernel lookup
    data_indices = np.arange(data.shape[0])
    
    # convert data to format used by kernel perceptron
    data_class = LabelledDataset(data)
    
    # generate dictionary of all kernel matrices with different parameters
    Kdict = make_kernel_dict(data_class.data, kernel_func, param_set)
    
    # store errors in dictionary
    error_dict = {}
    
    # loop through kernel parameters
    for d in tqdm(param_set, 'd'):
        
        # store train and test errors
        all_train_errors = np.zeros(runs)
        all_test_errors = np.zeros(runs)
        
        for this_run in tqdm(range(runs), 'run'):
            
            # split the dataset into train and test and store splitted indices
            train_set, test_set, train_indices, test_indices = train_test_split(data, data_indices, test_size=0.2)

            # convert data to format used by kernel perceptron
            train_set = LabelledDataset(train_set)
            test_set = LabelledDataset(test_set)
            
            # initialize kernel perceptron
            kp = Kernel_perceptron(train_set, 
                                   test_set, 
                                   train_indices, 
                                   test_indices, 
                                   nclasses = 10, 
                                   kernel_mtx = Kdict[str(d)], 
                                   kernel_param = d, 
                                   classification_method = classification_method)
            
            # train kernel perceptron
            train_error = kp.train(max_epochs=20)
            test_error = kp.test_error()
            
            all_train_errors[this_run] = train_error
            all_test_errors[this_run] = test_error
            
        # calculate means and standard deviations
        train_error_mean = np.mean(all_train_errors)
        train_error_std = np.std(all_train_errors)
        test_error_mean = np.mean(all_test_errors)
        test_error_std = np.std(all_test_errors)
        
        # store means and standard deviations in nested dictionary
        if str(d) not in error_dict:
            error_dict[str(d)] = {}
        error_dict[str(d)]['train_error_mean'] = train_error_mean
        error_dict[str(d)]['train_error_std'] = train_error_std
        error_dict[str(d)]['test_error_mean'] = test_error_mean
        error_dict[str(d)]['test_error_std'] = test_error_std
        
        print('d =', d)
        print('train error: %f ± %f' %(train_error_mean, train_error_std))
        print('test  error: %f ± %f\n' %(test_error_mean, test_error_std))
        
    return error_dict


def task2_perceptron(data, runs=20, kernel_func='polynomial', kval=5, classification_method='OvA', 
                     param_set=range(1,8), calc_confusion_mtx=False):
    """
    This function executes a kval-Fold cross-validation to tune
    the parameter 'd' of either a polynomial or a Gaussian kernel.
    The cross-validation is directly implementated withouth the use 
    of libraries. For computational efficiency, all kernel matrices 
    are computed at the beginning of the program, and indices are 
    used to lookup the kernel values.
    
    -- Input --
    data: [datasize, datadim] array -- all data
    runs: int -- number of runs to average over
    kernel_func: str -- 'polynomial' or 'Gaussian'
    kval: int -- number of folds in cross-validation
    classification_method: str -- 'OvA' (one vs all) or 'OvO' (one vs one)
    param_set: list or arr -- set containing all kernel parameters
    calc_confusion_mtx: Bool -- if set to True, a confusion matrix is outputted
    
    -- Return --
    best_d: float -- mean of best kernel parameter
    test_errors: list of floats -- errors on test set for every run
    mean_conf_mtx, std_conf_mtx: [nclass, nclass]-array -- mean and std of confusion mtx
    """
    # keep track of indices for kernel lookup
    data_indices = np.arange(data.shape[0])
    data_class = LabelledDataset(data)
    
    # generate dictionary of all kernel matrices with different parameters
    Kdict = make_kernel_dict(data_class.data, kernel_func, param_set)
    
    # save best d and corresponding test error of every run
    best_d = np.zeros(runs)
    test_errors = np.zeros(runs)
    
    # save confusion matrices
    all_conf_mtx = []
    
    for this_run in tqdm(range(runs), 'Run'):
        
        error_best_d = float('inf')
        this_best_d = 0
        
        # randomly split the dataset into .8 train and .2 test and store splitted indices
        train_set, test_set, train_indices, test_indices = train_test_split(data, data_indices, test_size=0.2)

        # split the random train set into kval-fold validation sets
        cross_val_sets = np.array_split(train_set, kval, axis=0)
        cross_val_indices = np.array_split(train_indices, kval, axis=0)
        
        # convert train and test set to input format of Kernel_perceptron
        train_set = LabelledDataset(train_set)
        test_set = LabelledDataset(test_set)
    
        for d in tqdm(param_set, 'd'):
            
            # sum the k-fold validation errors
            total_val_error = 0
            valset_count = 0
            
            for val_idx in range(kval):
                thisfold_train_set = cross_val_sets.copy()
                thisfold_train_indices = cross_val_indices.copy()
                
                # take the next fold as validation set
                val_set = thisfold_train_set.pop(val_idx)
                val_indices = thisfold_train_indices.pop(val_idx)
                
                # merge the other folds as training set
                thisfold_train_set = np.concatenate(thisfold_train_set)
                thisfold_train_indices = np.concatenate(thisfold_train_indices)

                # convert data sets to input format of Kernel_perceptron
                thisfold_train_set = LabelledDataset(thisfold_train_set)
                val_set = LabelledDataset(val_set)
                
                # initialize kernel perceptron
                thisfold_kp = Kernel_perceptron(thisfold_train_set, 
                                               val_set, 
                                               thisfold_train_indices, 
                                               val_indices, 
                                               nclasses = 10, 
                                               kernel_mtx = Kdict[str(d)], 
                                               kernel_param = d, 
                                               classification_method = classification_method)
                
                # train kernel perceptron and find train and test errors
                thisfold_train_error = thisfold_kp.train(max_epochs=20)
                thisfold_val_error = thisfold_kp.test_error()

                # add validation error to total
                total_val_error += (thisfold_val_error * val_set.size)
                valset_count += val_set.size
            
            # calculate weighted validation error from k-folds
            val_error = total_val_error / valset_count
            
            # check if this d is better
            if val_error < error_best_d:
                this_best_d = d
                error_best_d = val_error
                    
        # save best parameter d of this run
        best_d[this_run] = this_best_d        
        
        # train on full train set with best d
        kp = Kernel_perceptron(train_set, 
                               test_set, 
                               train_indices, 
                               test_indices, 
                               nclasses = 10, 
                               kernel_mtx = Kdict[str(d)], 
                               kernel_param = this_best_d, 
                               classification_method = classification_method)
        train_error = kp.train(max_epochs=20)
        test_errors[this_run] = kp.test_error()
        
        # produce confusion matrix
        if calc_confusion_mtx:
            conf_mtx = kp.confusion_matrix()
            all_conf_mtx.append(conf_mtx)
    
    print('Run', this_run+1)
    print('d^* mean: %f ± %f' %(np.mean(best_d), np.std(best_d)))
    print('test_err: %f ± %f' %(np.mean(test_errors), np.std(test_errors)))
    
    if calc_confusion_mtx:
        all_conf_mtx = np.array(all_conf_mtx)
        mean_conf_mtx = np.mean(all_conf_mtx, axis=0)
        std_conf_mtx = np.std(all_conf_mtx, axis=0)
    
        return best_d, test_errors, mean_conf_mtx, std_conf_mtx
    
    else:
        return best_d, test_errors
    
    
def five_hardest(data, runs=20, kernel_func='polynomial', kval=5, classification_method='OvA', 
                     param_set=[4,5]):
    """
    This function executes a kval-Fold cross-validation to find the
    five hardest data items to predict. The cross-validation is directly 
    implementated withouth the use of libraries. For computational 
    efficiency, all kernel matrices are computed at the beginning of 
    the program, and indices are used to lookup the kernel values.
    
    -- Input --
    data: [datasize, datadim] array -- all data
    runs: int -- number of runs to average over
    kernel_func: str -- 'polynomial' or 'Gaussian'
    kval: int -- number of folds in cross-validation
    classification_method: str -- 'OvA' (one vs all) or 'OvO' (one vs one)
    param_set: list or arr -- set containing all kernel parameters
    
    -- Return --
    hardest_five: (5,)-array -- array containing indices of 5 hardest to predict data items
    """
    
    mistake_vec = np.zeros(data.shape[0])
    
    # keep track of indices for kernel lookup    
    data_indices = np.arange(data.shape[0])
    data_class = LabelledDataset(data)
    
    # generate dictionary of all kernel matrices with different parameters
    Kdict = make_kernel_dict(data_class.data, kernel_func, param_set)
    
    for this_run in tqdm(range(runs), 'Run'):
        
        # randomly split the dataset into .8 train and .2 test and store splitted indices
        train_set, test_set, train_indices, test_indices = train_test_split(data, data_indices, test_size=0.2)

        # split the random train set into kval-fold validation sets
        cross_val_sets = np.array_split(train_set, kval, axis=0)
        cross_val_indices = np.array_split(train_indices, kval, axis=0)
        
        # convert train and test set to input format of Kernel_perceptron
        train_set = LabelledDataset(train_set)
        test_set = LabelledDataset(test_set)
    
        for d in tqdm(param_set, 'd'):
            
            # sum the k-fold validation errors
            total_val_error = 0
            valset_count = 0
            
            for val_idx in range(kval):
                thisfold_train_set = cross_val_sets.copy()
                thisfold_train_indices = cross_val_indices.copy()
                
                # take the next fold as validation set
                val_set = thisfold_train_set.pop(val_idx)
                val_indices = thisfold_train_indices.pop(val_idx)
                
                # merge the other folds as training set
                thisfold_train_set = np.concatenate(thisfold_train_set)
                thisfold_train_indices = np.concatenate(thisfold_train_indices)

                # convert data sets to input format of Kernel_perceptron
                thisfold_train_set = LabelledDataset(thisfold_train_set)
                val_set = LabelledDataset(val_set)
                
                # initialize kernel perceptron
                thisfold_kp = Kernel_perceptron(thisfold_train_set, 
                                               val_set, 
                                               thisfold_train_indices, 
                                               val_indices, 
                                               nclasses = 10, 
                                               kernel_mtx = Kdict[str(d)], 
                                               kernel_param = d, 
                                               classification_method = classification_method)
                
                # train kernel perceptron
                thisfold_train_error = thisfold_kp.train(max_epochs=20)
                
                mistake_vec += thisfold_kp.count_mistake_vec()

    
    hardest_five = np.argpartition(mistake_vec, -5)[-5:]
    
    return hardest_five
