from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from helpers import calculate_rmse_score

def lin_reg_blending(all_preds_train_trains, y_train_trains, all_preds_train_tests, y_train_tests, all_preds_tests, verbose=True):
    """
    Takes predictions on train, test and dataset_testing as 2d numpy array as well as scores on train and test set.
    Trains linear regression using predictions on train set, testing on test set.

    returns scikit learn model and predictions on dataset_testing
    """

    # initialize linear regressor
    lin_regr = LinearRegression()
    # fit using predictions train dataset
    lin_regr.fit(all_preds_train_trains, y_train_trains)
    # predict on train, test and dataset_testing datasets
    lin_regr_y_preds_trains_trains = lin_regr.predict(all_preds_train_trains)
    lin_regr_y_preds_trains_tests = lin_regr.predict(all_preds_train_tests)
    lin_regr_y_preds_tests = lin_regr.predict(all_preds_tests)

    # print if given True
    if verbose:
        # calculate tmse on train and test set
        rmse_train_train = calculate_rmse_score(lin_regr_y_preds_trains_trains, y_train_trains)
        rmse_train_test= calculate_rmse_score(lin_regr_y_preds_trains_tests, y_train_tests)
        print(rmse_train_train)
        print(rmse_train_test)

    # return model and predictions on dataset_testing
    return lin_regr, lin_regr_y_preds_tests

def ridge_reg_blending(all_preds_train_trains, y_train_trains, all_preds_train_tests, y_train_tests, all_preds_tests, alpha=2700000, verbose=True):
    """
    Takes predictions on train, test and dataset_testing as 2d numpy array as well as scores on train and test set.
    Trains ridge regression using predictions on train set, testing on test set.
    Alpha controls regularization, check scikit learn docs for more detail

    returns scikit learn model and predictions on dataset_testing
    """

    # initialize ridge regressor with given regularization
    ridge_regr = Ridge(alpha)
    # fit using predictions train dataset
    ridge_regr.fit(all_preds_train_trains, y_train_trains)
    # predict on train, test and dataset_testing datasets
    ridge_regr_y_preds_trains_trains = ridge_regr.predict(all_preds_train_trains)
    ridge_regr_y_preds_trains_tests = ridge_regr.predict(all_preds_train_tests)
    ridge_regr_y_preds_tests = ridge_regr.predict(all_preds_tests)

    # print if given True
    if verbose:
        # calculate tmse on train and test set
        rmse_train_train = calculate_rmse_score(ridge_regr_y_preds_trains_trains, y_train_trains)
        rmse_train_test = calculate_rmse_score(ridge_regr_y_preds_trains_tests, y_train_tests)
        print(rmse_train_train)
        print(rmse_train_test)

    # return model and predictions on dataset_testing
    return ridge_regr, ridge_regr_y_preds_tests

def nn_blending(all_preds_train_trains, y_train_trains, all_preds_train_tests, y_train_tests, all_preds_tests, max_iter=3, alpha=100, verbose=True):
    """
    Takes predictions on train, test and dataset_testing as 2d numpy array as well as scores on train and test set.
    Trains neural net regressor using predictions on train set, testing on test set.
    max_iter controls number of iterations, Alpha controls regularization, check scikit learn docs for more detail

    returns scikit learn model and predictions on dataset_testing
    """

    # initialize neural net regressor with given max_iter and regularization
    mlp = MLPRegressor(max_iter=max_iter, alpha=alpha, verbose=verbose)
    # fit using predictions train dataset
    mlp.fit(all_preds_train_trains, y_train_trains)
    # predict on train, test and dataset_testing datasets
    mlp_y_preds_trains_trains = mlp.predict(all_preds_train_trains)
    mlp_y_preds_trains_tests = mlp.predict(all_preds_train_tests)
    mlp_y_preds_tests = mlp.predict(all_preds_tests)

    # print if given True
    if verbose:
        # calculate tmse on train and test set
        rmse_train_train = calculate_rmse_score(mlp_y_preds_trains_trains, y_train_trains)
        rmse_train_test = calculate_rmse_score(mlp_y_preds_trains_tests, y_train_tests)
        print(rmse_train_train)
        print(rmse_train_test)

    # return model and predictions on dataset_testing
    return mlp, mlp_y_preds_tests

def save_model(clf, filename="ridge_regr"):
    """
    Takes sklearn model, saves it to disk
    """
    joblib.dump(clf, 'sklearn_models/{}.pkl'.format(filename))


def load_model(filename="ridge_regr"):
    """
    Loads sklearn model from disk
    """
    clf = joblib.load('sklearn_models/{}.pkl'.format(filename))

    return clf
