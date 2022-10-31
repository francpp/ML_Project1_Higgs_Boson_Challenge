""" Notebook for plots """
import numpy as np
from plot_utils import *
from initializers import load_csv_data, split_by_jet, pre_process_train
from implementations import ridge_regression

#%%
y, x, id_ = load_csv_data("train.csv")

#%%
plot_distribution_multivariates(y, x)

#%%
dict_train = split_by_jet (y, x)

###############################################################################
#%% Define the function to compute the prediction for each set
def prediction_for_roc(j, dict_train, lambda_, degree, quantile):
    y_train, x_train = dict_train[j]
    ind_train = np.array(np.where(x[:, 22] == j))
    tx_train, mean_train, std_train, med_train, quant_train = pre_process_train(x_train, degree = degree, 
                                                                          quantile = quantile)
    w, _ = ridge_regression(y_train, tx_train, lambda_ = lambda_) 
    fitted = tx_train@w
    
    return ind_train, fitted


#%% Set the parameters (different for each jet)
Lambda_ = [1e-05, 1e-05, 0.0001, 0.0001]
Degree = [5, 5, 5, 5]
Quantile = [0.925, 0.9125, 0.95, 0.925]

#%%
# Do the prediction
ids = []
fits = []

for j in range(4):   
    print('Running model', j)
    quantile = Quantile[j]
    lambda_ = Lambda_[j]
    degree = Degree[j]
    ind_train, fitted = prediction_for_roc(j, dict_train, 
                                      lambda_ = lambda_, degree = degree, 
                                      quantile = quantile)
    ids.append(ind_train.flatten())
    fits.append(fitted.flatten())

#%% ROC Curve
idd = np.concatenate(ids, axis = 0)
fitt = np.concatenate(fits, axis = 0)
AUC = Roc_curve(y[idd], fitt)

###############################################################################
#%% GRID SEARCH 
y0 = dict_train[0][0]
x0 = dict_train[0][1]

#%% (1) GS lambda
quantile = Quantile[0]
degree = Degree[0]
lambdas = [10**(-i) for i in range(1,10)]

ACC_TRAIN_L, ACC_TEST_L = grid_search_lambdas(y0, x0, quantile, lambdas, degree)
#%%
plot_grid_lambdas(lambdas, ACC_TRAIN_L, ACC_TEST_L)

#%% (2) GS quantile
lambda_ = Lambda_[0]
degree = Degree[0]
quantiles = [0.9, 0.925, 0.95, 0.975]

ACC_TRAIN_Q, ACC_TEST_Q = grid_search_quantile(y0, x0, quantiles, lambda_, degree)
#%%
plot_grid_quantile(quantiles, ACC_TRAIN_Q, ACC_TEST_Q)

#%% (3) GS degree
lambda_ = Lambda_[0]
degrees = [i for i in range(1, 7)]
quantile = Quantile[0]

ACC_TRAIN_D, ACC_TEST_D = grid_search_degree(y0, x0, quantile, lambda_, degrees)
#%%
plot_grid_degree(degrees, ACC_TRAIN_D, ACC_TEST_D)