import glob, os
import numpy as np
import torch
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

def train_load(filename):
    """
    takes filename, reads filename.csv in the same folder as code running
    assumes userID, itemID, rating columns are present
    """
    
    # set extension to .csv
    extension = '.csv'
    # set URL_PREFIX, set to "" to check the same folder
    URL_PREFIX = ''
    
    # read data using np.genfromtxt
    data = np.genfromtxt(URL_PREFIX + filename + extension, delimiter=',', names=True, dtype=(int, int, float))
    # take userID column
    users = data['userID']
    # take itemID column
    items = data['itemID']
    # take ratings column
    ratings = data['rating']
    
    # return in a tuple
    return (users, items, ratings)

def test_load(filename):
    """
    takes filename, reads filename.csv in the same folder as code running
    assumes userID, itemID columns are present
    """
    
    # set extension to .csv
    extension = '.csv'
    # set URL_PREFIX, set to "" to check the same folder
    URL_PREFIX = ''
    
    # read data using np.genfromtxt
    data = np.genfromtxt(URL_PREFIX + filename + extension, delimiter=',', names=True, dtype=(int, int))
    # take userID column
    users = data['userID']
    # take itemID column
    items = data['itemID']
    
    # return in a tuple
    return (users, items)

def get_train(myfile):
    """
    returns training set appropriate for spotlight.interactions.Interactions
    """
    
    # return training set as spotlight.interactions 
    return Interactions(*train_load(myfile))

def get_test(myfile):
    """
    returns testing set appropriate for spotlight.interactions.Interactions
    """
    
    # return testing set as spotlight.interactions 
    return Interactions(*test_load(myfile))
	
def train_spotlight_models(train, test, dataset_testing, embedding_dims, n_iters, batch_sizes, l2s, learning_rates, is_save = False):
    """
    takes train, test, dataset_testing datasets as spotlight.interactions.
    train multiple spotlight models using ExplicitFactorizationModel, with given parameters.
    parameters are given in embedding_dims, n_iters, batch_sizes, l2s, learning_rates.
    return predictions of train, test, dataset_testing datasets as well as rmse on train and test.
    """
    
    # initialize train_rmses and test_rmses, these store rmse on train and test set
    train_rmses = np.array([])
    test_rmses = np.array([])
    # initialize preds_train_trains, preds_train_tests, preds_tests; these store predictions of models  on train, test and dataset_testing datasets
    preds_train_trains = []
    preds_train_tests = []
    preds_tests = []
    
    # traverse all parameter combinations
    # embedding_din, n_iter, batch_size, l2 regularization, learning_rate 
    for embedding_dim in embedding_dims:
        for n_iter in n_iters:
            for batch_size in batch_sizes:
                for l2 in l2s:
                    for learning_rate in learning_rates:
                        # initialize model with parameter, ues GPU is torch.cuda.is_available() returns True, otherwise use CPU
                        model = ExplicitFactorizationModel(loss='regression',
                                                           embedding_dim=embedding_dim,  # latent dimensionality
                                                           n_iter=n_iter,  # number of epochs of training
                                                           batch_size=batch_size,  # minibatch size
                                                           l2=l2,  # strength of L2 regularization
                                                           learning_rate=learning_rate,
                                                           use_cuda=torch.cuda.is_available())
                        
                        # print which model is being trained
                        print("embedding_dim={}, n_iter={}, batch_size={}, l2={}, learning_rate={}".format(embedding_dim, n_iter, batch_size, l2, learning_rate))
                        # fit model
                        model.fit(train, verbose=True)
                        # find rmse on train
                        train_rmse = rmse_score(model, train)
                        # find rmse on test
                        test_rmse = rmse_score(model, test)
                        # store rmse on train and test sets
                        train_rmses = np.append(train_rmses, train_rmse)
                        test_rmses = np.append(test_rmses, test_rmse)   
                        # print train and test rmses
                        print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))
                        # if is_save given, save the models to disk
                        if is_save:
                            torch.save(model, "models/embedding_dim={}, n_iter={}, batch_size={}, l2={}, learning_rate={}".format(embedding_dim, n_iter, batch_size, l2, learning_rate))
                        # find predictions of train, test and dataset_testing datasets
                        preds_train_train = model.predict(train.user_ids,train.item_ids)
                        preds_train_test = model.predict(test.user_ids,test.item_ids)
                        preds_test = model.predict(dataset_testing.user_ids,dataset_testing.item_ids)
                        #store those predictions
                        preds_train_trains.append(preds_train_train)
                        preds_train_tests.append(preds_train_test)
                        preds_tests.append(preds_test)
    
    # return stored predictions on train, test, dataset_testing; return rmses on train and test
    return preds_train_trains, preds_train_tests, preds_tests, train_rmses, test_rmses

def train_spotlight_models_using_all_data(train, dataset_testing, embedding_dims, n_iters, batch_sizes, l2s, learning_rates, verbose=True):
    """
    takes train dataset as spotlight.interactions.
    train multiple spotlight models using ExplicitFactorizationModel, with given parameters.
    parameters are given in embedding_dims, n_iters, batch_sizes, l2s, learning_rates.
    saves trained models into disk
    """
    
    # store predictions on test set
    preds_tests = []

    # traverse all parameter combinations
    # embedding_din, n_iter, batch_size, l2 regularization, learning_rate 
    for embedding_dim in embedding_dims:
        for n_iter in n_iters:
            for batch_size in batch_sizes:
                for l2 in l2s:
                    for learning_rate in learning_rates:
                        # initialize model
                        model = ExplicitFactorizationModel(loss='regression',
                                                           embedding_dim=embedding_dim,  # latent dimensionality
                                                           n_iter=n_iter,  # number of epochs of training
                                                           batch_size=batch_size,  # minibatch size
                                                           l2=l2,  # strength of L2 regularization
                                                           learning_rate=learning_rate,
                                                           use_cuda=torch.cuda.is_available())
                        
                        # print if given True
                        if verbose:
                            print("embedding_dim={}, n_iter={}, batch_size={}, l2={}, learning_rate={}".format(embedding_dim, n_iter, batch_size, l2, learning_rate))
                        # fit model using train dataset
                        model.fit(train, verbose=verbose)
                        preds_test = model.predict(dataset_testing.user_ids,dataset_testing.item_ids)
                        preds_tests.append(preds_test)
                        # save model to disk
                        torch.save(model, "models_all_data/embedding_dim={}, n_iter={}, batch_size={}, l2={}, learning_rate={}".format(embedding_dim, n_iter, batch_size, l2, learning_rate))

    # return stored predictions on dataset_testing
    return preds_tests

def load_spotlight_models(train, test, dataset_testing, verbose=False):
    """
    Loads pretrained spotlight models from the folder in the directory. 
    Takes train, test datasets and dataset_testing to generate predictions and calculate rmse
    """
    
    # initialize predictions, stores predictions on train, test and dataset_testing datasets
    preds_train_trains = []
    preds_train_tests = []
    preds_tests = []
    # initialize rmses, stores rmses on train and test dataset
    train_rmses = np.array([])
    test_rmses = np.array([])
    
    # for each file in the "models" folder in the directory
    for file in glob.glob("models/*"):
        # prinr filenames, if given True
        if verbose:
            print(file)
        # load model
        model = torch.load(file)
        # calculate and store rmses on train and test datasets
        train_rmse = rmse_score(model, train)
        test_rmse = rmse_score(model, test)
        train_rmses = np.append(train_rmses, train_rmse)
        test_rmses = np.append(test_rmses, test_rmse)
        # make predictions on train, test and dataset_testing datasets
        preds_train_train = model.predict(train.user_ids,train.item_ids)
        preds_train_test = model.predict(test.user_ids,test.item_ids)
        preds_test = model.predict(dataset_testing.user_ids,dataset_testing.item_ids)
        # store predictions
        preds_train_trains.append(preds_train_train)
        preds_train_tests.append(preds_train_test)
        preds_tests.append(preds_test)
    
    # return predictions on train, test and dataset_testing datasets; return rmse on train and test datasets
    return preds_train_trains, preds_train_tests, preds_tests, train_rmses, test_rmses
	
def load_spotlight_models_using_all_data(dataset_testing, verbose=False):
    """
    Loads pretrained spotlight models from the folder in the directory. 
    Takes dataset_testing to generate predictions
    """
    
    # initialize predictions, stores predictions on dataset_testing datasets
    preds_tests = []
    
    # for each file in the "models" folder in the directory
    for file in glob.glob("models_all_data/*"):
        # print filenames, if given True
        if verbose:
            print(file)
        # load model
        model = torch.load(file)
        preds_test = model.predict(dataset_testing.user_ids,dataset_testing.item_ids)
        # store predictions
        preds_tests.append(preds_test)
    
    # return predictions on train, test and dataset_testing datasets; return rmse on train and test datasets
    return preds_tests
	
def get_scores(train, test):
    """
    Takes train and test datasets, returns scores
    """
    
    # take scores
    y_train_trains = train.ratings
    y_train_tests = test.ratings
    
    # return score
    return y_train_trains, y_train_tests