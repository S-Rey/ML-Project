import numpy as np
import csv

def calculate_rmse_score(x1, x2):
    """
    Calculates rmse score between given two numpy vectors of same size
    """
    
    # return rmse between x1, x2
    return np.sqrt(np.mean(np.square(x1-x2)))
	
def filter_models(test_rmses, preds_train_trains, preds_train_tests, preds_tests, threshold=1.2):
    """
    Takes predictions on train, test and dataset_testing; as well as test_rmses and threshold
    
    returns predictions with test rmse smaller than given threshold
    """
    
    # find index of models with smaller test rmse on threshold
    index_lst = list(np.where(test_rmses < threshold)[0])
    
    # initialize predictions
    preds_train_trains_best = []
    preds_train_tests_best = []
    preds_tests_best = []
    
    # traverse each prediction
    for i in range(len(preds_tests)):
        # check if test rmse smaller than given threshold
        if i in index_lst:
            # append predictions
            preds_train_trains_best.append(preds_train_trains[i])
            preds_train_tests_best.append(preds_train_tests[i])
            preds_tests_best.append(preds_tests[i])
            
    # return filtered predictions
    return preds_train_trains_best, preds_train_tests_best, preds_tests_best
	
def stack_predictions(preds_train_trains, preds_train_tests, preds_tests):
    """
    Given predictions as a list of numpy arrays, stack them together.
    
    return 2d numpy array for predictions on train, test and dataset_testing 
    where number of columns are number of different predictions
    """
    
    # stack predictions
    all_preds_train_trains = np.vstack(preds_train_trains).T
    all_preds_train_tests = np.vstack(preds_train_tests).T
    all_preds_tests = np.vstack(preds_tests).T
    
    # return stacked predictions
    return all_preds_train_trains, all_preds_train_tests, all_preds_tests

def create_csv_submission(user_ids, movie_ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    
    # open file as csvfile
    with open(name, 'w') as csvfile:
        # set fieldnames
        fieldnames = ['Id', 'Prediction']
        # write
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        # write predictions
        for u, m, pred in zip(user_ids, movie_ids, y_pred):
            writer.writerow({'Id':"r{}_c{}".format(u,m),'Prediction':float(pred)})
