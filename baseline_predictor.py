import numpy as np
from helpers import calculate_rmse_score

class Predictor:
    """
    Base class for predictors
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # base function
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.scores = scores
        self.params_dict = params_dict

    def train(self):
        # base train
        pass

    def predict(self, test_user_ids, test_item_ids):
        # base predict
        pass

class GlobalMeanPredictor(Predictor):
    """
    GlobalMeanPredictor, predicts global mean of ratings given for each user, item pair
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # initialize predictor, set given parameters
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_mean = None

    def train(self):
        # find global mean
        self.global_mean = np.mean(self.scores)

    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_mean not known, find
        if self.global_mean is None:
            self.train()
            
        # set predictions to global_mean
        test_scores_preds = self.global_mean * np.ones(test_user_ids.shape[0])
        
        # save if given
        if is_save:
            np.save("global_mean_predictions"+filename, test_scores_preds)
           
        # return predictions
        return test_scores_preds
    
class GlobalMedianPredictor(Predictor):
    """
    GlobalMedianPredictor, predicts global median of ratings given for each user, item pair
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # initialize predictor, set given parameters
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_median = None
        
    def train(self):
        # find global median
        self.global_median = np.median(self.scores)
    
    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_median not known, find
        if self.global_median is None:
            self.train()
            
        # set predictions to global_median
        test_scores_preds = self.global_median * np.ones(test_user_ids.shape[0])
        
        # save if given
        if is_save:
            np.save("global_median_predictions"+filename, test_scores_preds)
        
        # return predictions
        return test_scores_preds
    
class UserMeanPredictor(Predictor):
    """
    UserMedianPredictor, predicts mean of ratings for given user, if new user is asked predicts global mean
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # initialize predictor, set given parameters
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_mean = None
        
    def train(self):
        # find global mean
        self.global_mean = np.mean(self.scores)
    
    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_mean not known, find
        if self.global_mean is None:
            self.train()
            
        # set predictions to global_mean
        test_scores_preds = self.global_mean * np.ones(test_user_ids.shape[0])
        
        # for each user
        for user_tmp in list(set(test_user_ids)):
            # if user in user_ids given before
            if user_tmp in self.user_ids:
                # set prediction for that user as mean rating for that user
                test_scores_preds[np.where(test_user_ids == user_tmp)] = np.mean(self.scores[np.where(self.user_ids == user_tmp)])
        
        # save if given
        if is_save:
            np.save("user_mean_predictions"+filename, test_scores_preds)
            
        # return predictions
        return test_scores_preds
    
class ItemMeanPredictor(Predictor):
    """
    ItemMeanPredictor, predicts mean of ratings for given item, if new item is asked predicts global mean
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # initialize predictor, set given parameters
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_mean = None
        
    def train(self):
        # find global mean
        self.global_mean = np.mean(self.scores)
    
    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_mean not known, find
        if self.global_mean is None:
            self.train()
            
        # set predictions to global_mean
        test_scores_preds = self.global_mean * np.ones(test_item_ids.shape[0])
        
        # for each item
        for item_tmp in list(set(test_item_ids)):
            # if item in item_ids given before
            if item_tmp in self.item_ids:
                # set prediction for that item as mean rating for that item
                test_scores_preds[np.where(test_item_ids == item_tmp)] = np.mean(self.scores[np.where(self.item_ids == item_tmp)])
        
        # save if given
        if is_save:
            np.save("item_mean_predictions"+filename, test_scores_preds)
            
        # return predictions
        return test_scores_preds
    
class UserMedianPredictor(Predictor):
    """
    UserMedianPredictor, predicts median of ratings for given user, if new user is asked predicts global median
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # initialize predictor, set given parameters
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_median = None
        
    def train(self):
        # find global median
        self.global_median = np.median(self.scores)
    
    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_median not known, find
        if self.global_median is None:
            self.train()
            
        # set predictions to global_median
        test_scores_preds = self.global_median * np.ones(test_user_ids.shape[0])
        
        # for each user
        for user_tmp in list(set(test_user_ids)):
            # if user in user_ids given before
            if user_tmp in self.user_ids:
                # set prediction for that user as median rating for that user
                test_scores_preds[np.where(test_user_ids == user_tmp)] = np.median(self.scores[np.where(self.user_ids == user_tmp)])
        
        # save if given
        if is_save:
            np.save("user_median_predictions"+filename, test_scores_preds)
        
        # return predictions
        return test_scores_preds
    
class ItemMedianPredictor(Predictor):
    """
    ItemMedianPredictor, predicts median of ratings for given item, if new item is asked predicts global median
    """
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        # initialize predictor, set given parameters
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_median = None
        
    def train(self):
        # find global median
        self.global_median = np.median(self.scores)
    
    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_median not known, find
        if self.global_median is None:
            self.train()
            
        # set predictions to global_median
        test_scores_preds = self.global_median * np.ones(test_item_ids.shape[0])
        
        # for each item
        for item_tmp in list(set(test_item_ids)):
            # if item in item_ids given before
            if item_tmp in self.item_ids:
                # set prediction for that item as median rating for that item
                test_scores_preds[np.where(test_item_ids == item_tmp)] = np.median(self.scores[np.where(self.item_ids == item_tmp)])
        
        # save if given
        if is_save:
            np.save("item_median_predictions"+filename, test_scores_preds)
        
        # return predictions
        return test_scores_preds
    
class UserItemMoodPredictor(Predictor):
    """
    UserItemMoodPredictor, predicts (user/item) (mean/median) + (item/user) mood based on (mean/median)
                            if new (user/item) is asked predicts global (mean/median)
                            
    for example,
    user mood based on mean = mean of ratings for that user - mean of mean of ratings for each user
    """
    # initialize predictor, set given parameters
    def __init__(self, user_ids, item_ids, scores, params_dict={}):
        Predictor.__init__(self, user_ids, item_ids, scores, params_dict)
        self.global_mean = None
        self.global_median = None
        
    def train(self):
        # find global mean and median
        self.global_mean = np.mean(self.scores)
        self.global_median = np.median(self.scores)
    
    def predict(self, test_user_ids, test_item_ids, is_save=False, filename=""):
        # if global_mean or global_median not known, find
        if self.global_mean is None or self.global_median is None:
            self.train()
            
        # set initial predictions
        user_means = self.global_mean * np.ones(test_user_ids.shape[0])
        item_means = self.global_mean * np.ones(test_item_ids.shape[0])
        user_medians = self.global_median * np.ones(test_user_ids.shape[0])
        item_medians = self.global_median * np.ones(test_item_ids.shape[0])
        
        # for each user
        for user_tmp in list(set(test_user_ids)):
            # if user in user_ids given before
            if user_tmp in self.user_ids:
                # set prediction for that user as mean rating for that user
                user_means[np.where(test_user_ids == user_tmp)] = np.mean(self.scores[np.where(self.user_ids == user_tmp)])
                # set prediction for that user as median rating for that user
                user_medians[np.where(test_user_ids == user_tmp)] = np.median(self.scores[np.where(self.user_ids == user_tmp)])
        
        # for each item
        for item_tmp in list(set(test_item_ids)):
            # if item in item_ids given before
            if item_tmp in self.item_ids:
                # set prediction for that item as mean rating for that item
                item_means[np.where(test_item_ids == item_tmp)] = np.mean(self.scores[np.where(self.item_ids == item_tmp)])
                # set prediction for that item as median rating for that item
                item_medians[np.where(test_item_ids == item_tmp)] = np.median(self.scores[np.where(self.item_ids == item_tmp)])
        
        # find mean of mean ratings of users
        global_user_mean = np.mean(user_means)
        # find mean of mean ratings of items
        global_item_mean = np.mean(item_means)
        # find median of median ratings of users
        global_user_median = np.median(user_medians)
        # find median of median ratings of items
        global_item_median = np.median(item_medians)
                
        # find user moods based on mean
        user_mean_moods = user_means - global_user_mean
        # find item moods based on mean
        item_mean_moods = item_means - global_item_mean
        # find user moods based on median
        user_median_moods = user_medians - global_user_median
        # find item moods based on median
        item_median_moods = item_medians - global_item_median
        
        # calculate 8 different predictions
        # using 2 (user/item) * 2 (mean/median) * 2 (mean_mood/median_mood) combinations
        user_mean_item_mean_mood = user_means + item_mean_moods
        user_median_item_mean_mood = user_medians + item_mean_moods
        user_mean_item_median_mood = user_means + item_median_moods
        user_median_item_median_mood = user_medians + item_median_moods
        item_mean_user_mean_mood = item_means + user_mean_moods
        item_median_user_mean_mood = item_medians + user_mean_moods
        item_mean_user_median_mood = item_means + user_median_moods
        item_median_user_median_mood = item_medians + user_median_moods
        
        # save if given
        if is_save:
            np.save("user_mean_item_mean_mood"+filename, user_mean_item_mean_mood)
            np.save("user_median_item_mean_mood"+filename, user_median_item_mean_mood)
            np.save("user_mean_item_median_mood"+filename, user_mean_item_median_mood)
            np.save("user_median_item_median_mood"+filename, user_median_item_median_mood)
            np.save("item_mean_user_mean_mood"+filename, item_mean_user_mean_mood)
            np.save("item_median_user_mean_mood"+filename, item_median_user_mean_mood)
            np.save("item_mean_user_median_mood"+filename, item_mean_user_median_mood)
            np.save("item_median_user_median_mood"+filename, item_median_user_median_mood)
        
        # return predictions
        return user_mean_item_mean_mood, user_median_item_mean_mood, user_mean_item_median_mood, user_median_item_median_mood, item_mean_user_mean_mood, item_median_user_mean_mood, item_mean_user_median_mood, item_median_user_median_mood 
		
def train_baseline_models(train, test, dataset_testing):
    """
    Trains baseline models.
    Train models on train; evaluate on train and test; store predictions on train, test and dataset_testing
    """
    
    # initialize models
    g_mean_prd = GlobalMeanPredictor(train.user_ids, train.item_ids, train.ratings)
    g_median_prd = GlobalMedianPredictor(train.user_ids, train.item_ids, train.ratings)
    u_mean_prd = UserMeanPredictor(train.user_ids, train.item_ids, train.ratings)
    i_mean_prd = ItemMeanPredictor(train.user_ids, train.item_ids, train.ratings)
    u_median_prd = UserMedianPredictor(train.user_ids, train.item_ids, train.ratings)
    i_median_prd = ItemMedianPredictor(train.user_ids, train.item_ids, train.ratings)
    mood_prd = UserItemMoodPredictor(train.user_ids, train.item_ids, train.ratings)
    
    # make list of models
    models = [mood_prd, g_mean_prd, g_median_prd, u_mean_prd, i_mean_prd, u_median_prd, i_median_prd]
    
    # initialize predictions on train, test and dataset_testing datasets
    baseline_preds_train_trains = []
    baseline_preds_train_tests = []
    baseline_preds_tests = []
    # initialize rmses on train and test datasets
    baseline_train_rmses = []
    baseline_test_rmses = []
    
    # traverse models list
    for model in models:
        # make predictions on train, test and dataset_testing datasets
        train_train_pred = model.predict(train.user_ids, train.item_ids, filename="train_train")
        train_test_pred = model.predict(test.user_ids, test.item_ids, filename="train_test")
        test_pred = model.predict(dataset_testing.user_ids, dataset_testing.item_ids, filename="test")
        
        # if tuple is returned, handle it differently (for handling UserItemMoodPredictor)
        if type(train_train_pred) is tuple:
            # for each of predictions
            for i in range(len(train_train_pred)):
                # store predictions on train, test and dataset_testing datasets by appending
                baseline_preds_train_trains.append(train_train_pred[i])
                baseline_preds_train_tests.append(train_test_pred[i])
                baseline_preds_tests.append(test_pred[i])
                # calculate and store rmses on train and test datasets
                baseline_train_rmses.append(calculate_rmse_score(train_train_pred[i], train.ratings))
                baseline_test_rmses.append(calculate_rmse_score(train_test_pred[i], test.ratings))
                
        # if single prediction is returned
        else:
            # calculate and store rmses on train and test datasets
            baseline_train_rmses.append(calculate_rmse_score(train_train_pred, train.ratings))
            baseline_test_rmses.append(calculate_rmse_score(train_test_pred, test.ratings))
            # store predictions on train, test and dataset_testing datasets by appending
            baseline_preds_train_trains.append(train_train_pred)
            baseline_preds_train_tests.append(train_test_pred)
            baseline_preds_tests.append(test_pred)
    
     # return predictions on train, test and dataset_testing datasets; return rmse on train and test datasets
    return baseline_preds_train_trains, baseline_preds_train_tests, baseline_preds_tests, baseline_train_rmses, baseline_test_rmses
