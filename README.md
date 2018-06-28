# Machine Learning Project 2

Collaborators:
 * [Furkan](https://github.com/afofa)
 * [RadoslawDryzner](https://github.com/RadoslawDryzner)
 * [S-Rey](https://github.com/S-Rey)

### Synopsis
This repository contains all the code of the project of the Machine Learning class given at [EPFL](http://www.epfl.ch).
The purpose of this project was to create a recommender system and to test it on the 
"Netflix Prize" dataset. A report discussing about the results is also available in this 
repository.


### Running it


The data is available [here](https://transfer.sh/cmbnh/data.zip). "data" folder should be in the same older as run.py.

We split the data into two. In order to make sure split is the same we set the seed. The models that are trained for that particular split are available [here](https://transfer.sh/AnCn3/models.zip) (if you change the seed, you should train the models from scratch). We also train models on whole dataset, those models are [here](https://transfer.sh/G6Cse/models_all_data.zip). These are models from library called spotlight, it has pytorch backend. These models are trained on a single GPU in 2 hours. We successfully loaded these models and get predictions using CPU and GPU.

If you have problem any problem with pretrained spotlight models, we also saved predictions. The predictions for models that are trained on split are [here](https://transfer.sh/aEGl8/models_numpy.zip). The predictions for models that are trained on whole training dataset are [here](https://transfer.sh/d3Lxv/models_all_data_numpy.zip).

"models", "models_all_data", "models_numpy", "models_all_data_numpy", "data" folders should be in the same directory as run.py.
All other python scripts ("helpers.py", "spotlight_helpers.py", "surprise_helpers.py", "baseline_predictor.py", "blending.py") should also be in the same directory as run.py.

Data, models and scripts can also be retrieved from [here](https://drive.google.com/open?id=1ZITo0C4kBKOxKIncVecwyhAAFGEeX9rd).

There are two boolean variables in the code that controls this flow.
* is_train_spotlight = True, train all the models from scratch. if GPU is available, uses GPU (~2hours on our computer). otherwise uses CPU (no time estimation).
* is_train_spotlight = False and is_load_from_numpy = True, loads saved predictions. just trains simple ridge regression. shouldn't take more than several minutes.
* is_train_spotlight = False and is_load_from_numpy = False, loads saved spotlight models and makes predictions again. can take 5-10 minutes.
