"""Define the functions useful to do the cross validation of different models"""

import numpy as np
from helpers import best_threshold, fit_model, calculate_acc_F1, sigmoid
from initializers import pre_process_train, pre_process_test
from implementations import *


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, quantile, lambda_, degree, fun = 'ridge'):
    """"Apply the cross validation to a model splitted in k-folds using the
    k-fold as a test set"""
    k_train = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    k_test = k_indices[k]
    
    x_test = x[k_test, :]
    y_test = y[k_test]
    x_train = x[k_train,:]
    y_train = y[k_train]
    
    tx_train, mean_train, std_train, med_train, quant_train = pre_process_train(x_train, degree = degree, quantile = quantile)
    tx_test = pre_process_test(x_test, mean_train, std_train, med_train, quant_train,
                               degree = degree, quantile = quantile)
    np.random.seed(100)
    initial_w = np.zeros((tx_train.shape[1], 1))
    
    if fun=='mse_gd':
        w, _ = mean_squared_error_gd(y_train, tx_train, initial_w = initial_w, 
                                     max_iters = 1000, gamma = 0.001)
    elif fun=='mse_sgd':
        w, _ = mean_squared_error_sgd(y_train, tx_train, initial_w = initial_w,
                                      max_iters = 1000, gamma = 0.001)
    elif fun=='ls':
        w, _ = least_squares(y_train, tx_train)
    elif fun=='ridge':
        w, _ = ridge_regression(y_train, tx_train, lambda_)
    elif fun=='log':
        w, _ = logistic_regression(y_train, tx_train, initial_w = initial_w, 
                                   max_iters = 1, gamma = 0.001)
    elif fun=='log_ridge':
        w, _ = reg_logistic_regression(y_train, tx_train, lambda_, initial_w = initial_w,
                                       max_iters = 5, gamma = 0.0001)
    
    val = 0
    acc_tr = 0
    fitted = tx_test@w
    if(fun == 'log' or fun =='log_ridge'):        
        val, acc_tr = best_threshold(y_train, tx_train, w, model = 'logistic', visualize = False)
        fitted = sigmoid(tx_test@w)
    else:
        val, acc_tr = best_threshold(y_train, tx_train, w, model = 'LS', visualize = False)
    
    y_test_fitted = fit_model(fitted, threshold = val)
    acc_te, _ = calculate_acc_F1 (y_test, y_test_fitted)
    
    return w, acc_tr, acc_te
