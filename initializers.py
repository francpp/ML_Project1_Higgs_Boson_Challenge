"""Some initializers functions, useful to load the dataset and create the 
final feature matrices"""

import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def sample_data(y, x, seed, size_samples):
    """Sample from dataset"""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices, :]
    return y[:size_samples], x[:size_samples]

def split_by_jet (y, tx, idx = 0, inp = 'train'):
    """Split the dataset according to the PRI_jet_num" and return a dictionary
    where the key is the PRI_jet_num"""
    diction = {}
    if inp == 'train':       
        for j in range(4):
            index = np.where(tx[:, 22] == j)
            diction[j] = (y[index], np.delete(tx[index], 22, axis =1))
    else:
        for j in range(4):
            index = np.where(tx[:, 22] == j)
            diction[j] = (idx[index], np.delete(tx[index], 22, axis =1))
    return diction

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degree = 1, bias = False):
    """Build polynomials augmenting each features until the value of degree; 
    if selected, add the bias column"""
    x_new = np.ones((x.shape[0], 1))
    cols = []
    cols.append(x_new)
    for col in range(x.shape[1]):
        powers = np.arange(1, degree+1)[:, None]
        mat = (x[:, col]**powers).T
        cols.append(mat)
    x_new = np.concatenate(cols, axis = 1)
    if not bias:    
        x_new = x_new[:, 1:]
    return x_new

def feature_mix(x, degree):
    """Add to the feature matrix the mixed-polynomials and the bias"""
    x1 = np.ones((x.shape[0], 1))
    lis = [x1]
    lis.append(x)
    feature = 0
    for i in range(x.shape[1]):
        for j in range((feature+1)*degree, x.shape[1]):
            col = (x[:, i]*x[:, j]).reshape(-1, 1)
            lis.append(col)
        if (i == degree*(feature+1)-1):
            feature+=1
    x = np.concatenate(lis, axis = 1)
    return x

def pre_process_train(x, degree, quantile, deleteCols = True):
    """Return the preprocessed train feature matrix (with the other parameters useful 
    to do the same transformations into the test), given as input the initial matrix 
    and some hyperparameters"""   
    # Delete features with useless columns
    if deleteCols:
        cols_to_del = [15, 18, 20, 24, 27]
        x = np.delete(x, cols_to_del, axis = 1)
    
    # Delete the features with lots of NaN
    perc_val_ok = np.count_nonzero(((x != -999) & (x != 0)), axis = 0)/x.shape[0]
    x = x[:, (perc_val_ok > 0.75)]
    
    # Replace missing values (of the first feature) ​​with the median
    x_0 = x[:, 0]
    med_x0 = np.median(x_0[x_0!=-999])
    x[x==-999] = med_x0
    
    # Cap values using the quantile
    quant_ = np.quantile(x, quantile, axis = 0)
    x = x*(x<=quant_)+quant_*(x>quant_)
    
    # Log-transform and standardize
    x = np.log(x+1-np.min(x, axis = 0))
    
    # Augment the dataset
    tx = build_poly(x, degree = degree, bias = False)
    tx = feature_mix(tx, degree = degree)   
    
    tx[:,1:], mean_x, std_x = standardize(tx[:, 1:])
    
    return tx, mean_x, std_x, med_x0, quant_
    
def pre_process_test(x, mean_train, std_train, med_train, quant_train,
                     degree, quantile, deleteCols = True):
    """Return the preprocessed test feature matrix, given as input the initial 
    matrix, the values coming from input and some hyperparameters"""   
    # Delete features with useless columns
    if deleteCols:
        cols_to_del = [15, 18, 20, 24, 27]
        x = np.delete(x, cols_to_del, axis = 1)
    
    # Delete the features with lots of NaN
    perc_val_ok = np.count_nonzero(((x != -999) & (x != 0)), axis = 0)/x.shape[0]
    x = x[:, (perc_val_ok > 0.75)]
    
    # Replace missing values (of the first feature) ​​with the median
    x[x==-999] = med_train
    
    # Cap values using the quantile
    x = x*(x<=quant_train)+quant_train*(x>quant_train)
    
    # Log-transform and standardize
    x = np.log(x+1-np.min(x, axis = 0))
    
    # Augment the dataset
    tx = build_poly(x, degree = degree, bias = False)
    tx = feature_mix(tx, degree = degree)    
    
    tx[:, 1:] = (tx[:, 1:] - mean_train)/std_train
    return tx