import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LabelledDataset():
    
    def __init__(self, dataset):
        self.data = dataset[:,1:]
        self.ndim = self.data.shape[1]
        self.size = self.data.shape[0]
        self.labels = dataset[:,0]
        
               
def make_kernel_dict(data, kernel_func, param_set):
    """
    Create dictionary containing polynomial or Gaussian kernel matrices 
    with different polynomial parameter or Gaussian parameters respectively.
    
    -- Input --
    data: [datasize, datadim] array -- all data
    kernel_func: str -- 'polynomial' or 'Gaussian'
    param_set: list or arr -- set containing all kernel parameters
    
    -- Return --
    Kdict: dictionary -- { parameter: [datasize, datasize]-kernel_matrix }
    """
    
    Kdict = {}
    
    # check if valid kernel function-name
    if kernel_func != 'polynomial' and kernel_func != 'Gaussian':
            print("WRONG KERNEL FUNCTION!")
            
    # calculate polynomial kernel matrix for different parameters
    if kernel_func == 'polynomial':
        for param in param_set:
            Kdict[str(param)] = np.power(np.dot(data, np.transpose(data)), param)
            
    # calculate Gaussian kernel matrix for different parameters
    elif kernel_func == 'Gaussian':
        dist = cdist(data, data, 'euclidean')
        for param in param_set:
            Kdict[str(param)] = np.exp(-param * np.power(dist, 2))
            
    return Kdict
               
def draw_heatmap(mean_mtx, std_mtx):
    """
    Draw heatmap for confusion matrix
    """
               
    # dimensions of confusion matrix
    dims = mean_mtx.shape
    
    # create labels
    labels = np.empty(dims, dtype=np.dtype('U12'))
    for row in range(dims[0]):
        for col in range(dims[1]):
            labels[row, col] = str(np.around(100*mean_mtx[row, col], 3)) + "%\n± " + str(np.around(100*std_mtx[row, col], 3))

    # draw figure
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(mean_mtx, cmap='Wistia')
        
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    plt.xticks(np.arange(10), fontsize=16)
    plt.yticks(np.arange(10), fontsize=16)
    
    plt.xlabel('Predicted label', fontsize=18)
    plt.ylabel('True label', fontsize=18)    

    # draw labels
    for (j,i), label in np.ndenumerate(labels):
        ax.text(i,j, label, ha='center', va='center', fontsize=14)
    plt.show()
    
    
def show_hardest_five(hardest_5_idx):
    """
    Show instances that are hardest to predict
    """
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    for idx, h in enumerate(hardest_5_idx):
        ax[idx].imshow(data[h,1:].reshape(16,16), cmap='gray', vmin=-1, vmax=1)
        ax[idx].set_title('Label: ' + str(int(data[h,0])), fontsize=20)
        ax[idx].axis('off')
    plt.show()
