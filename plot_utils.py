"""Functions called to generate plots"""
import matplotlib.pyplot as plt
import numpy as np
from helpers import fit_model
from cross_val import cross_validation, build_k_indices

###############################################################################
###############################################################################
def grid_search_lambdas(y, tx, quantile, lambdas, degree):
    k_fold = 4
    k_indices = build_k_indices(y, k_fold, seed = 100)
    
    ACC_TRAIN_BEST = np.zeros(len(lambdas))
    ACC_TEST_BEST = np.zeros(len(lambdas))
        
    for i, lambda_ in enumerate(lambdas): 
        print('lambda_ =', lambda_)           
        acc_train = []
        acc_test = []               
        for k in range (k_fold):
            w, acc_tr, acc_te = cross_validation(y, tx, k_indices, k,
                                                 quantile = quantile,
                                                 lambda_ = lambda_,
                                                 degree = degree)          
            acc_train.append(acc_tr)
            acc_test.append(acc_te)            
        acc_train_best = np.mean(acc_train)
        acc_test_best = np.mean(acc_test)        

        ACC_TRAIN_BEST[i] = acc_train_best
        ACC_TEST_BEST[i] = acc_test_best        
    
    return ACC_TRAIN_BEST, ACC_TEST_BEST

def plot_grid_lambdas(lambdas, ACC_TRAIN_BEST, ACC_TEST_BEST):
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.figure()
    plt.semilogx(lambdas, ACC_TRAIN_BEST, 'r-x', label = 'train')
    plt.semilogx(lambdas, ACC_TEST_BEST, 'k-x', label = 'test')
    plt.xlabel('lambda')
    plt.ylabel('accuracy')
    plt.xlim(np.max(lambdas), np.min(lambdas))
    plt.grid() 
    plt.title('Grid search for lambda\n PRI_jet_num = 0')
    plt.legend()
    plt.savefig(f"GSlambda.png", bbox_inches = "tight")

def grid_search_quantile(y, tx, quantiles, lambda_, degree):
    k_fold = 4
    k_indices = build_k_indices(y, k_fold, seed = 100)
    
    ACC_TRAIN_BEST = np.zeros(len(quantiles))
    ACC_TEST_BEST = np.zeros(len(quantiles))
        
    for i, quantile in enumerate(quantiles): 
        print('quantile =', quantile)           
        acc_train = []
        acc_test = []               
        for k in range (k_fold):
            w, acc_tr, acc_te = cross_validation(y, tx, k_indices, k,
                                                 quantile = quantile,
                                                 lambda_ = lambda_,
                                                 degree = degree)          
            acc_train.append(acc_tr)
            acc_test.append(acc_te)            
        acc_train_best = np.mean(acc_train)
        acc_test_best = np.mean(acc_test)        

        ACC_TRAIN_BEST[i] = acc_train_best
        ACC_TEST_BEST[i] = acc_test_best        
    
    return ACC_TRAIN_BEST, ACC_TEST_BEST

def plot_grid_quantile(quantiles, ACC_TRAIN_BEST, ACC_TEST_BEST):
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.figure()
    plt.plot(quantiles, ACC_TRAIN_BEST, 'r-x', label = 'train')
    plt.plot(quantiles, ACC_TEST_BEST, 'k-x', label = 'test')
    plt.xlabel('quantile')
    plt.ylabel('accuracy')
    plt.xlim(np.min(quantiles), np.max(quantiles))
    plt.grid()
    plt.title('Grid search for quantile\n PRI_jet_num = 0')
    plt.legend()
    plt.savefig(f"GSquantile.png", bbox_inches = "tight")

    
def grid_search_degree(y, tx, quantile, lambda_, degrees):
    k_fold = 4
    k_indices = build_k_indices(y, k_fold, seed = 100)
    
    ACC_TRAIN_BEST = np.zeros(len(degrees))
    ACC_TEST_BEST = np.zeros(len(degrees))
        
    for i, degree in enumerate(degrees): 
        print('degree =', degree)           
        acc_train = []
        acc_test = []               
        for k in range (k_fold):
            w, acc_tr, acc_te = cross_validation(y, tx, k_indices, k,
                                                 quantile = quantile,
                                                 lambda_ = lambda_,
                                                 degree = degree)          
            acc_train.append(acc_tr)
            acc_test.append(acc_te)            
        acc_train_best = np.mean(acc_train)
        acc_test_best = np.mean(acc_test)        

        ACC_TRAIN_BEST[i] = acc_train_best
        ACC_TEST_BEST[i] = acc_test_best        
    
    return ACC_TRAIN_BEST, ACC_TEST_BEST

def plot_grid_degree(degrees, ACC_TRAIN_BEST, ACC_TEST_BEST):
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.figure()
    plt.plot(degrees, ACC_TRAIN_BEST, 'r-x', label = 'train')
    plt.plot(degrees, ACC_TEST_BEST, 'k-x', label = 'test')
    plt.xlabel('degree')
    plt.ylabel('accuracy')
    plt.xlim(np.min(degrees), np.max(degrees))
    plt.grid()
    plt.title('Grid search for degree\n PRI_jet_num = 0')
    plt.legend()
    plt.savefig(f"GSdegree.png", bbox_inches = "tight")

###############################################################################
###############################################################################
def plot_predictions(y, x, labels = (1, 2)):    
    colors = np.array(['red', 'green'])[y]
    plt.scatter(x[:, labels[0]], x[:, labels[1]], c=colors, s=10)
    plt.show()

def plot_distribution_multivariates(yy, xx):
    import seaborn as sns
    labels = list[range(xx.shape[1])]
    sns.set_style('whitegrid')
    plt.rcParams["figure.figsize"] = (6,4)

    for idx, lab in enumerate (labels):
        print(idx, lab)
        if idx == 1 or idx == 15:
            if idx ==1: 
                name = "DER_mass_transverse_met_lep"
            if idx == 15:
                name= "PRI_lep_phi"
            quantile = np.quantile(xx[:,lab], 0.995)
            xx[ xx[:,lab] > quantile, lab] = quantile
            
            x_zero, x_one = xx[yy == 0, lab], xx[yy ==1, lab]
            x_zero, x_one = x_zero[x_zero != -999], x_one[x_one != -999]
            
            
            
            sns.kdeplot(x_zero, color = "red", label = "y = 0")
            sns.kdeplot(x_one, color = "black", label = "y = 1")
            plt.xlabel("Values")
            plt.legend()
            plt.title(f"Distribution of {name}")
            plt.savefig(f"Distribution of {name}.png", bbox_inches = "tight")
            plt.show()

###############################################################################
###############################################################################
def Roc_curve(y, fitted):
    hp_threshold = np.linspace(np.min(fitted), np.max(fitted), 1001)
    fpr = []
    tpr = []

    for t in hp_threshold:
        y_fitted = fit_model(fitted, threshold = t)
        TP=(np.count_nonzero((y == 1) & (y_fitted == 1))/len(y))
        FP=(np.count_nonzero((y == 0) & (y_fitted == 1))/len(y))
        FN=(np.count_nonzero((y == 1) & (y_fitted == 0))/len(y))
        TN=(np.count_nonzero((y == 0) & (y_fitted == 0))/len(y))
        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN)) 
    
    print(np.max(fpr))
    plt.figure()
    plt.grid()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    diff = np.flip(fpr[:1000]-fpr[1:])
    tpr = np.flip(tpr)
    tpr = (tpr[:1000]+tpr[1:])/2
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    AUC = np.sum(tpr*diff)
    plt.title(f'ROC curve\nAUC: {np.round(AUC,4)}')
    plt.show()
    return AUC
    
    
def plot_corr_matrix(xx):
    plt.imshow(np.corrcoef(xx.T))
    plt.colorbar()
    plt.title('Correlation matrix')
    plt.show()
    
