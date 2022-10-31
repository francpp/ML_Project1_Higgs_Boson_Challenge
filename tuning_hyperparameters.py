"""Tune the hyperparameters (quantile, lambda_, degree) combining grid search 
and cross validation"""

#%% import libraries
import numpy as np
from initializers import load_csv_data, split_by_jet
from cross_val import build_k_indices, cross_validation

#%% Load data
y, x, id_ = load_csv_data("train.csv")

#%% Split by jet
dict_train = split_by_jet (y, x)

#%% Define the grid search
lambdas = [10**(-i) for i in range(3, 7)]
degrees = [1, 2, 3, 4, 5]
quantiles = [0.90, 0.9125, 0.925, 0.95]
k_fold = 4

#%% Search the best hyperparameters for every PRI_jet_num
train_accu = []
test_accu = []

for j in range(4):
    print()
    print('Running jet:', j)
    y, tx = dict_train[j]
    y = y.reshape(-1, 1)
    k_indices = build_k_indices(y, k_fold, seed = 100)
    
    ACC_TRAIN_BEST = np.zeros((len(quantiles), len(lambdas), len(degrees)))
    ACC_TEST_BEST = np.zeros((len(quantiles), len(lambdas), len(degrees)))
        
    for q, quantile in enumerate(quantiles):        
        for i, lambda_ in enumerate(lambdas):
            for j, degree in enumerate(degrees):
                
                acc_train = []
                acc_test = []
                       
                for k in range (k_fold):
                    w, acc_tr, acc_te = cross_validation(y, tx, k_indices, k,
                                                         quantile = quantile,
                                                         lambda_ = lambda_,
                                                         degree = degree,
                                                         fun = 'ridge')          
                    acc_train.append(acc_tr)
                    acc_test.append(acc_te)
                    
                acc_train_best = np.mean(acc_train)
                acc_test_best = np.mean(acc_test)
                
                print('quantile =', quantile, 'lambda_ =', lambda_, 'degree =', degree)
                print('acc_train =', acc_train_best, 'acc_test =', acc_test_best)
                ACC_TRAIN_BEST[q, i, j] = acc_train_best
                ACC_TEST_BEST[q, i, j] = acc_test_best
                
                quantile_index, lambda_index, degree_index = np.unravel_index(ACC_TEST_BEST.argmax(), 
                                                            ACC_TEST_BEST.shape)
        
    train_accu.append(ACC_TRAIN_BEST[ quantile_index, lambda_index, degree_index])
    test_accu.append(ACC_TEST_BEST[ quantile_index, lambda_index, degree_index])
    

        
    print(quantiles[quantile_index], lambdas[lambda_index], degrees[degree_index])   

print('Train accuracy = ',train_accu)
print('Test accuracy = ',test_accu)

#%%    
def weighted_average(y, test, train, dict_):
    w_test, w_train = 0 , 0
    for ind, j in enumerate(dict_.keys()):
        weight = dict_[j][0].shape[0] / len(id_)
        #print(weight)
        w_test += weight * test[ind]
        w_train += weight * train[ind]
    return w_test, w_train


test, train = weighted_average(y, test_accu, train_accu, dict_train)
print(f"Test Accuracy: {np.round(test, 5)}, Train Accuracy: {np.round(train, 5)}")
