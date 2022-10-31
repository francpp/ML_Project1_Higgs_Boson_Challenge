import numpy as np

###############################################################################
###############################################################################
""" The 6 implemented methods """

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    loss, w = gradient_descent(y, tx, initial_w, max_iters, gamma)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    loss, w = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)
    return w, loss

def least_squares(y, tx):
    tx_t = tx.transpose()
    w = np.linalg.solve(tx_t @ tx, tx_t @ y)
    loss = compute_loss(y, tx, w, loss_type='mse')
    return w, loss

def ridge_regression(y, tx, lambda_):
    D = tx.shape[1]
    lambda_star = lambda_*2*len(y)
    A = np.transpose(tx).dot(tx) + lambda_star*np.eye(D)
    b = np.transpose(tx).dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w, loss_type='mse')
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    loss, w = logistic_regression_SGD(y, tx, initial_w, max_iters, gamma = gamma, 
                                      batch_size = len(y))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    loss, w = logistic_regression_ridge(y, tx, initial_w, max_iters, lambda_ = lambda_, 
                                        gamma = gamma, batch_size = len(y))
    return w, loss

###############################################################################
###############################################################################
"""Define the learning step in agree with every method""" 

def learning_step(y, tx, w, lambda_, gamma, loss_type):  
    gradient = compute_gradient(y, tx, w, lambda_, loss_type)
    w = w - gamma*gradient
    loss = compute_loss(y, tx, w, loss_type)
    return loss, w

###############################################################################
"""Define the gradient descent algorithms"""

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = compute_loss(y, tx, w, loss_type = 'mse')
    for n_iter in range(max_iters):
        loss, w = learning_step(y, tx, w, lambda_ = 0, gamma = gamma, loss_type = 'mse')
    return loss, w

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for yb, txb in batch_iter(y, tx, batch_size, int(len(y)/batch_size)):          
            loss, w = learning_step(yb, txb, w, lambda_ = 0, gamma = gamma, loss_type = 'mse')  
    return loss, w

def logistic_regression_SGD(y, tx, initial_w, max_iters,
                            gamma, batch_size):
    np.random.seed(100)
    w = initial_w.reshape(-1, 1)
    losses = []
    losses.append(compute_loss(y, tx, w, loss_type = 'log'))
    for n_iter in range(max_iters):
        loss_iter = []
        for yb, txb in batch_iter(y, tx, batch_size, num_batches = int(1)):
            loss, w = learning_step(yb, txb, w, lambda_ = 0, gamma = gamma, loss_type = 'log')
            losses.append(loss)
            loss_iter.append(loss)            
    loss = losses[-1]
    return loss, w 

def logistic_regression_ridge(y, tx, initial_w, max_iters, lambda_,
                            gamma, batch_size):
    np.random.seed(100)
    w = initial_w.reshape(-1, 1)
    losses = []
    losses.append(compute_loss(y, tx, w, loss_type = 'log_ridge'))
    for n_iter in range(max_iters):
        loss_iter = []
        for yb, txb in batch_iter(y,tx, batch_size = batch_size, 
                                  num_batches = int(len(y)/batch_size)):
            loss, w = learning_step(yb, txb, w, lambda_ = lambda_, 
                                    gamma = gamma, loss_type = 'log_ridge')
            losses.append(loss)
            loss_iter.append(loss)        
    loss = losses[-1]
    return loss, w 

###############################################################################
###############################################################################
"""Compute the gradients for the different loss functions"""

def compute_gradient(y, tx, w, lambda_, loss_type = 'mse'):  
    if loss_type == 'mse':
        return compute_gradient_mse(y, tx, w)
    elif loss_type == 'log':
        return compute_gradient_logistic(y, tx, w)
    elif loss_type == 'log_ridge':
        return compute_gradient_logistic_ridge(y, tx, w, lambda_)

###############################################################################

def compute_gradient_mse(y, tx, w):
    err = y-tx@w
    n = len(y)
    gradient = -tx.T@err/n
    return gradient 

def compute_gradient_logistic(y, tx, w):
    sig_fitted = sigmoid(tx @ w)
    gradient = tx.T@(sig_fitted-y)/len(y)
    return gradient

def compute_gradient_logistic_ridge(y, tx, w, lambda_):
    sig_fitted = sigmoid(tx @ w)
    gradient = tx.T@(sig_fitted-y)/len(y) + 2*lambda_*w
    return gradient

###############################################################################
###############################################################################
"""Compute the costs for the different loss functions"""

def compute_loss(y, tx, w, loss_type = 'mse'):  
    if loss_type == 'mse':
        return compute_mse(y, tx, w)
    elif (loss_type == 'log' or loss_type == 'log_ridge'):
        return compute_loss_logistic(y, tx, w)
    
###############################################################################

def compute_mse(y, tx, w):
    N = y.shape[0]
    return np.sum((y-tx@w)**2)/(2*N)

def compute_loss_logistic(y, tx, w):
    sig_fitted = sigmoid(tx @ w)
    losses =  y.T@(np.log(sig_fitted)) + (1-y).T@(np.log(1 - sig_fitted))
    return - losses/len(y)

###############################################################################
###############################################################################
"""Define two helper function: the batch iterator and the sigmoid function"""

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

def sigmoid(x):
    """Return the sigmoid of a point, useful for logistic regression"""
    return 1/(1+np.exp(-x))