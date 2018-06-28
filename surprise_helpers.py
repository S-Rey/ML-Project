import surprise

def surprise_get_predictions(model, user_ids, item_ids):
    """
    Takes surprise model, list of user_ids and item_ids.
    Returns prediction of model for each of user_id, item_id pairs
    """
    
    # initialize list of predictions
    scores_pred = []
    
    # traverse each pair 
    for i in range(len(user_ids)):
        # find and store predictions
        scores_pred.append(model.predict(user_ids[i], item_ids[i]).est)
        
    # return as numpy array
    return np.array(scores_pred)

def train_surprise_baseline_models(df_train, df_test, df_dataset_testing):
    """
    Given train, test and dataset_testing datasets as dataframe, trains models available in surprise library
    """
    
    # makesure columns are in correct order
    df_train_train = df_train[["userID","itemID","rating"]]
    df_train_test = df_test[["userID","itemID","rating"]]
    df_test = df_dataset_testing[["userID","itemID"]]
    
    # initialize reader object, make sure rating are inbetween 1 and 5
    reader = surprise.Reader(rating_scale=(1, 5))
    # load dataset
    data_train_train = surprise.Dataset.load_from_df(df_train_train, reader)
    # make full trainset (check docs of surprise library for detail)
    data_train_train_trainset = data_train_train.build_full_trainset()
    
    
    print("normal Predictor")
    # initialize predictor
    normal_prd = surprise.prediction_algorithms.random_pred.NormalPredictor()
    normal_prd.train(data_train_train_trainset)

    bsl_options = {
        "method":"als",
        "reg_u":1,
        "reg_i":1,
        "n_epochs":10,
    }

    bsl_options = {
        "method":"sgd",
        "reg":10,
        "learning_rate":0.1,
        "n_epochs":10,
    }
    
    print("baseline Predictor")
    baseline_prd = surprise.prediction_algorithms.baseline_only.BaselineOnly(bsl_options)
    baseline_prd.train(data_train_train_trainset)

    sim_options = {
        "name":"pearson", #"cosine", "msd", "pearson"
        "user_based":False,
    }
    
    print("knnbasic Predictor")
    knnbasic_prd = surprise.prediction_algorithms.knns.KNNBasic(k=10, min_k=1, sim_options=sim_options)
    knnbasic_prd.train(data_train_train_trainset)
    
    print("knnwithmeans Predictor")
    knnwithmeans_prd = surprise.prediction_algorithms.knns.KNNWithMeans(k=10, min_k=1, sim_options=sim_options)
    knnwithmeans_prd.train(data_train_train_trainset)
    
    print("knnwithzscore Predictor")
    knnwithzscore_prd = surprise.prediction_algorithms.knns.KNNWithZScore(k=10, min_k=1, sim_options=sim_options)
    knnwithzscore_prd.train(data_train_train_trainset)
    
    print("knnbaseline Predictor")
    knnbaseline_prd = surprise.prediction_algorithms.knns.KNNBaseline(k=10, min_k=1, sim_options=sim_options, bsl_options=bsl_options)
    knnbaseline_prd.train(data_train_train_trainset)
    
    print("slopeone Predictor")
    slopeone_prd = surprise.prediction_algorithms.slope_one.SlopeOne()
    slopeone_prd.train(data_train_train_trainset)

    print("coclustering Predictor")
    coclustering_prd = surprise.prediction_algorithms.co_clustering.CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=5, verbose=True)
    coclustering_prd.train(data_train_train_trainset)
    
    models = [normal_prd, baseline_prd, knnbasic_prd, knnwithmeans_prd, knnwithzscore_prd, knnbaseline_prd, slopeone_prd, coclustering_prd]

    surprise_baseline_preds_train_trains = []
    surprise_baseline_preds_train_tests = []
    surprise_baseline_preds_tests = []
    surprise_baseline_train_rmses = []
    surprise_baseline_test_rmses = []
    
    for model in models:
        print(model, "predicting")
        train_train_pred = surprise_get_predictions(model, list(df_train_train.userID), list(df_train_train.itemID))
        train_test_pred = surprise_get_predictions(model, list(df_train_test.userID), list(df_train_test.itemID))
        test_pred = surprise_get_predictions(model, list(df_dataset_testing.userID),list(df_dataset_testing.itemID))

        surprise_baseline_train_rmses.append(calculate_rmse_score(train_train_pred, list(df_train_train.rating)))
        surprise_baseline_test_rmses.append(calculate_rmse_score(train_test_pred, list(df_train_test.rating)))
        surprise_baseline_preds_train_trains.append(train_train_pred)
        surprise_baseline_preds_train_tests.append(train_test_pred)
        surprise_baseline_preds_tests.append(test_pred)
        
    return models, surprise_baseline_preds_train_trains, surprise_baseline_preds_train_tests, surprise_baseline_preds_tests, surprise_baseline_train_rmses, surprise_baseline_test_rmses