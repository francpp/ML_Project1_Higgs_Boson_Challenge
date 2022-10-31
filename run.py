"""Here we run our final model, after an accurate tuning of the hyperparameters"""

#%% import libraries
from initializers import load_csv_data, split_by_jet, pre_process_train, pre_process_test
from implementations import ridge_regression
from helpers import best_threshold, pred_test
from generate_submission import submission

#%% Load data
y, x, _ = load_csv_data("train.csv")
y_test, x_test, id_test = load_csv_data("test.csv")

#%% Split by PRI_jet_num
dict_train = split_by_jet (y, x)
dict_test = split_by_jet (y_test, x_test, id_test, inp = 'test')

del y, x, y_test, x_test

#%% Define the function to compute the prediction for each set
def prediction(j, dict_train, dict_test, lambda_, degree, quantile):
    """ Given as input the PRI_jet_num, the splitted train and test set, and 
    the hypermeters, return the index and the prediction of the specific jet"""
    
    # Select the jet
    y_train, x_train = dict_train[j]
    idx_test, x_test = dict_test[j]
    
    # Pre-process the train and the test
    tx_train, mean_train, std_train, med_train, quant_train = pre_process_train(x_train, degree = degree, 
                                                                          quantile = quantile)
    tx_test = pre_process_test(x_test, mean_train, std_train, med_train, 
                               quant_train, degree = degree, quantile = quantile)
    
    # Run the specific algorithm to obtain the weights
    w, _ = ridge_regression(y_train, tx_train, lambda_ = lambda_) 
    
    # Choose the best_threshold for the classification
    val, _ = best_threshold(y_train, tx_train, w)
    
    # Predict the result in the test set
    y_pred_jet = pred_test(tx_test, w, val = val)
    
    return idx_test, y_pred_jet


#%% Set the parameters (different for each jet)
Lambda_ = [1e-05, 1e-05, 0.0001, 0.0001]
Degree = [5, 5, 5, 5]
Quantile = [0.925, 0.9125, 0.95, 0.925]

#%% Do the prediction
ids = []
y_pred = []

for j in range(4):   
    print('Running model PRI_jet_num = ', j)
    quantile = Quantile[j]
    lambda_ = Lambda_[j]
    degree = Degree[j]
    idx_test, y_pred_jet = prediction(j, dict_train, dict_test, 
                                      lambda_ = lambda_, degree = degree, 
                                      quantile = quantile)
    ids.append(idx_test)
    y_pred.append(y_pred_jet)

#%% Submit the prediction
SUBM_PATH = 'FinalSubmission/submission_end.csv'
submission(ids, y_pred, SUBM_PATH)