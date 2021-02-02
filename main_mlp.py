import numpy as np
from numpy.random import randn
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from MLP import *
from helper import *


data = pd.read_table("data/zipcombo.dat", sep="\s+")
data = np.array(data)


def task1_MLP(data, runs=20, param_set=[512, 256, 128, 64, 32]):
    """
    This function trains a Multi-Layer Perceptron (MLP) for different
    parameter values and takes the average over multiple runs.
    
    -- Input --
    data: [datasize, datadim] array -- all data
    runs: int -- number of runs to average over
    param_set: list or arr -- set containing mlp parameters
    
    -- Return --
    error_dict: dict -- dictionary containing { parameter: 
                                                { train_error: , train_std:, test_error: , test_std: } }
    """
    
    # convert data to format used by kernel perceptron
    data_class = LabelledDataset(data)
    
    # store errors in dictionary
    error_dict = {}
    
    # loop through kernel parameters
    for param in param_set:
        
        units_per_layer = [param, 10]
        
        # store train and test errors
        all_train_errors = np.zeros(runs)
        all_test_errors = np.zeros(runs)
        
        for this_run in range(runs):
            
            # split the dataset into train and test and store splitted indices
            train_set, test_set = train_test_split(data, test_size=0.2)

            # convert data to format used by kernel perceptron
            train_set = LabelledDataset(train_set)
            test_set = LabelledDataset(test_set)
            
            # initialize kernel perceptron
            mlp = MLP(train_set.ndim, units_per_layer)
            
            # train mlp
            mlp.train(train_set.data.T, 
                      one_hot(train_set.labels.T), 
                      test_set.data.T, 
                      one_hot(test_set.labels.T), verbose=500, batch_size=2000, epochs=2000, lr=.1)
            
            train_error, test_error = mlp.calc_errors(train_set.data.T, one_hot(train_set.labels.T), test_set.data.T, one_hot(test_set.labels.T))
            
            all_train_errors[this_run] = train_error
            all_test_errors[this_run] = test_error
            
        # calculate means and standard deviations
        train_error_mean = np.mean(all_train_errors)
        train_error_std = np.std(all_train_errors)
        test_error_mean = np.mean(all_test_errors)
        test_error_std = np.std(all_test_errors)
        
        # store means and standard deviations in nested dictionary
        keyname = str(param)
        if keyname not in error_dict:
            error_dict[keyname] = {}
        error_dict[keyname]['train_error_mean'] = train_error_mean
        error_dict[keyname]['train_error_std'] = train_error_std
        error_dict[keyname]['test_error_mean'] = test_error_mean
        error_dict[keyname]['test_error_std'] = test_error_std
        
        print('parameters =', param)
        print('train error: %f ± %f' %(train_error_mean, train_error_std))
        print('test  error: %f ± %f\n' %(test_error_mean, test_error_std))
        print('train error: %f' %(train_error_mean))
        print('test  error: %f\n' %(test_error_mean))
        
    return error_dict, mlp


def task2_mlp(data, runs=20, kval=5, param_set=[512, 384, 256, 128]):
    """
    This function executes a kval-Fold cross-validation to tune
    the number of units in the hidden layer of a MLP
    
    -- Input --
    data: [datasize, datadim] array -- all data
    runs: int -- number of runs to average over
    kval: int -- number of folds in cross-validation
    param_set: list or arr -- set containing all units
    
    -- Return --
    best_params: float -- mean of best kernel parameter
    test_errors: list of floats -- errors on test set for every run
    mlp
    """
    
    # save best d and corresponding test error of every run
    best_param = np.zeros(runs)
    test_errors = np.zeros(runs)
    
    for this_run in tqdm(range(runs), 'Run'):
        
        error_best_param = float('inf')
        this_best_param = 0
        
        # randomly split the dataset into .8 train and .2 test and store splitted indices
        train_set, test_set, = train_test_split(data, test_size=0.2)

        # split the random train set into kval-fold validation sets
        cross_val_sets = np.array_split(train_set, kval, axis=0)
        
        # convert train and test set to input format of Kernel_perceptron
        train_set = LabelledDataset(train_set)
        test_set = LabelledDataset(test_set)
    
        for param in tqdm(param_set, 'Param'):
            
            units_per_layer = [param, 10]
            
            # sum the k-fold validation errors
            total_val_error = 0
            valset_count = 0
            
            for val_idx in range(kval):
                thisfold_train_set = cross_val_sets.copy()
                
                # take the next fold as validation set
                val_set = thisfold_train_set.pop(val_idx)
                
                # merge the other folds as training set
                thisfold_train_set = np.concatenate(thisfold_train_set)

                # convert data sets to input format of Kernel_perceptron
                thisfold_train_set = LabelledDataset(thisfold_train_set)
                val_set = LabelledDataset(val_set)
                
                # initialize mlp
                mlp = MLP(train_set.ndim, units_per_layer)
                
                # train mlp
                mlp.train(thisfold_train_set.data.T, 
                          one_hot(thisfold_train_set.labels.T), 
                          val_set.data.T, 
                          one_hot(val_set.labels.T), batch_size=2000, epochs=1000, lr=.1)

                thisfold_train_error, thisfold_val_error = mlp.calc_errors(thisfold_train_set.data.T, 
                                                                           one_hot(thisfold_train_set.labels.T), 
                                                                           val_set.data.T, 
                                                                           one_hot(val_set.labels.T))

                # add validation error to total
                total_val_error += (thisfold_val_error * val_set.size)
                valset_count += val_set.size
            
            # calculate weighted validation error from k-folds
            val_error = total_val_error / valset_count
            
            # check if this d is better
            if val_error < error_best_param:
                this_best_param = param
                error_best_param = val_error
                    
        # save best parameter d of this run
        best_param[this_run] = this_best_param      
        
        # train on full train set with best d
        mlp = MLP(train_set.ndim, [this_best_param, 10])
        mlp.train(thisfold_train_set.data.T, 
                  one_hot(thisfold_train_set.labels.T), 
                  val_set.data.T, 
                  one_hot(val_set.labels.T), batch_size=2000, epochs=1000, lr=.1)
        
        train_error, test_error = mlp.calc_errors(train_set.data.T, 
                                                   one_hot(train_set.labels.T), 
                                                   test_set.data.T, 
                                                   one_hot(test_set.labels.T))
        test_errors[this_run] = test_error

    
    print('Run', this_run+1)
    print('d^* mean: %f ± %f' %(np.mean(best_param), np.std(best_param)))
    print('test_err: %f ± %f' %(np.mean(test_errors), np.std(test_errors)))
    
    return best_param, test_errors, mlp
