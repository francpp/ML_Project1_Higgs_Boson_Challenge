"""Some helper functions"""

import numpy as np

def sigmoid(x):
    """Return the sigmoid of a point, useful for logistic regression"""
    return 1/(1+np.exp(-x))

def fit_model(fitted, threshold = 0.5):
    """Return the label of the fitted values"""
    y_fitted = (fitted > threshold).flatten()
    return y_fitted*1

def pred_test(tx_test, w, val = 0.5):
    """Return the label obtained in the test set"""
    fitted_jet = tx_test@w
    y_pred_jet = np.copy(fitted_jet)
    y_pred_jet[fitted_jet > val] = 1
    y_pred_jet[fitted_jet <= val] = -1
    return y_pred_jet

def calculate_acc_F1 (y, y_fitted):
    """Compute the accuracy and the F1-score of a model"""
    y = y.reshape(-1)
    y_fitted = y_fitted.reshape(-1)
    TP = np.count_nonzero((y == 1) & (y_fitted == 1))
    TN = np.count_nonzero((y == 0) & (y_fitted == 0))
    FP = np.count_nonzero((y == 0) & (y_fitted == 1))
    FN = np.count_nonzero((y == 1) & (y_fitted == 0))
    acc = (TP+TN)/(TP+TN+FP+FN)
    
    F1 = 0
    if(TP != 0):
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = (2*precision*recall)/(precision+recall)
    return acc, F1

def best_threshold(y_train, x_train, w, model = 'LS', visualize = True):
    """Compute the best threshold for a model after calculating the weights"""
    if model == 'LS':
        fitted = x_train@w
    elif model == 'logistic':
        fitted = sigmoid(x_train@w) 
    
    hp_threshold = np.linspace(0.4, 0.6, 20)
    max_ = 0
    val = 0
    f_max = 0
    for t in hp_threshold:
        y_fitted = fit_model(fitted, threshold = t)
        acc, F1 = calculate_acc_F1(y_train, y_fitted)
        if acc > max_:
            max_ = acc
            val = t
            f_max = F1
    if visualize:    
        print('thr =', val, 'acc =', max_, 'F1 =', f_max)
    return val, max_
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset"""
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
