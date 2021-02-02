import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

    
class kNN:
    
    def __init__(self, train_set, test_set, k):
        
        self.train_set = train_set
        self.test_set = test_set
        self.k = k
        
        
    def calc_distance(self, test_points):
        """
        Calculate all Eucledian distances between training set and 'test_points'.
        """
        
        # Eucl dist= x^2 + y^2 - 2xy
        dists = np.sum(np.power(self.train_set.data, 2), axis=1) \
                + np.sum(np.power(test_points,2), axis=1)[:, np.newaxis] \
                -2 * np.dot(test_points, np.transpose(self.train_set.data))
                
        return dists
                
    def get_kNN(self, test_points, k):
        """
        Finds the k-nearest neighbours and return their labels.
        """
        
        # calculate all distances between training and 'test_points'
        dists = self.calc_distance(test_points)
        
        # sort array to find k-nearest neighbours
        NNs_idx = np.argsort(dists, axis=1)[:,:k]

        # find labels of k-nearest neighbours
        NNs_labels = self.train_set.labels[NNs_idx]
                
        return NNs_labels


    def predict(self, test_points, k):
        """
        Predict function
        """
        NNs_labels = self.get_kNN(test_points, k)
        
        predictions = []
        for this_dat, this_label in enumerate(NNs_labels):
            
            uniq_vals, counts = np.unique(this_label, return_counts=True)
            
            if np.count_nonzero(counts == counts.max()) == 1:
                idx = np.argmax(counts)
                predictions.append(uniq_vals[idx])
            
            else:
                temp_k = k - 1
                this_pred = self.predict(np.array([test_points[this_dat]]), temp_k)
                predictions.append(np.asscalar(this_pred))
        
        return np.array(predictions)
    
    
    def accuracy(self, test_points, labels):
        """ Calculate the accuracy """
        
        predictions = self.predict(test_points, self.k)
        
        mistakes = np.sum(predictions!=labels)

        return mistakes/labels.shape[0]
        
        
