# imports
import pandas as pd
from baseline_predictor import *
from blending import *
from helpers import *
from spotlight_helpers import *
from surprise_helpers import *


# load training data
dataset_training = get_train('data/training_data')
# cast ratings to np.float32, required bu spotlight library
dataset_training.ratings = dataset_training.ratings.astype(np.float32)
# load testing data
dataset_testing = get_test('data/testing_data')

# make sure dataset is correct
print("Training dataset: ", dataset_training)
print("Testing dataset: ", dataset_testing)

# set test_percentage
test_percentage = 0.1
# set random_seed
random_seed = 42
# split dataset_training into 2, train and test according to given test percentage and random seed
train, test = random_train_test_split(dataset_training, test_percentage=test_percentage, random_state=np.random.RandomState(random_seed))
# make sure splitted data is correct
print('Split into \n {} and \n {}.'.format(train, test))


# set if train spotlight models
is_train_spotlight = False

# load predictions from numpy array
is_load_from_numpy = True


# train spotlight models?
if is_train_spotlight:
        # parameters for spotlight models 
        embedding_dims = [8, 16, 32, 64]
        n_iters = [3, 4, 6]
        batch_sizes = [128, 512]
        l2s = [0, 1e-8, 1e-6, 1e-4]
        learning_rates = [1e-4, 1e-3]

        # train spotlight models
        print("Training spotlight models")
        sl_preds_train_trains, sl_preds_train_tests, sl_preds_tests, sl_train_rmses, sl_test_rmses = train_spotlight_models(train, test, dataset_testing, embedding_dims, n_iters, batch_sizes, l2s, learning_rates, is_save=False)

# load spotlight models?
else:
        # load from numpy?
        if is_load_from_numpy:
                # load numpy prediction directly
                print("Loading numpy predictions")
                sl_preds_train_trains = list(np.load("models_numpy/sl_preds_train_trains.npy"))
                sl_preds_train_tests = list(np.load("models_numpy/sl_preds_train_tests.npy"))
                sl_preds_tests = list(np.load("models_numpy/sl_preds_tests.npy"))
        else:
                # load spotlight models
                print("Loading spotlight models")
                sl_preds_train_trains, sl_preds_train_tests, sl_preds_tests, sl_train_rmses, sl_test_rmses = load_spotlight_models(train, test, dataset_testing, verbose=False)


# make 2d np arrays
all_preds_train_trains = np.vstack([sl_preds_train_trains]).T
all_preds_train_tests = np.vstack([sl_preds_train_tests]).T
all_preds_tests = np.vstack([sl_preds_tests]).T


# get ratings
y_train_trains, y_train_tests = get_scores(train, test)

# train blending, use ridge regression
ridge_regr, ridge_regr_y_preds_tests = ridge_reg_blending(all_preds_train_trains, y_train_trains, all_preds_train_tests, y_train_tests, all_preds_tests, alpha=2700000, verbose=True)



# train spotlight models?
if is_train_spotlight:
        # parameters for spotlight models 
        embedding_dims = [8, 16, 32, 64]
        n_iters = [3, 4, 6]
        batch_sizes = [128, 512]
        l2s = [0, 1e-8, 1e-6, 1e-4]
        learning_rates = [1e-4, 1e-3]

        # train spotlight models
        print("Training spotlight models using all of training set")
        preds_tests_all_data = train_spotlight_models_using_all_data(train, dataset_testing, embedding_dims, n_iters, batch_sizes, l2s, learning_rates, verbose=True)

# load spotlight models?
else:
        # load from numpy?
        if is_load_from_numpy:
                # load numpy prediction directly
                print("Loading numpy predictions")
                preds_tests_all_data = np.load("models_all_data_numpy/sl_preds_tests_all_data.npy")
        else:
                # load spotlight models
                print("Loding spotlight models using all of training set")
                preds_tests_all_data = list(load_spotlight_models_using_all_data(dataset_testing, verbose=False))

# make 2d np array
all_preds_tests_all_data = np.vstack([preds_tests_all_data]).T

# predict on test set
ridge_regr_y_preds_tests_all_data = ridge_regr.predict(all_preds_tests_all_data)
# generate submission.csv
print("Generating submission.csv")
create_csv_submission(dataset_testing.user_ids+1, dataset_testing.item_ids+1, np.clip(ridge_regr_y_preds_tests_all_data, 1, 5), "submission.csv")
