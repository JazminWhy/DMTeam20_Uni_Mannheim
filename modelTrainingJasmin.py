import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors.nearest_centroid import NearestCentroid

def select_model():
    pass

def naive_bayes(data, target, test):
    #naive_bayes = GaussianNB()
    #naive_bayes.fit(golf_encoded, golf['Play'])
    pass

def nearest_centroid():
    # nearest_centroid_estimator = NearestCentroid()
    # nearest_centroid_estimator.fit....
    # result_arr_knn = knn_estimator.predict(test)
    pass

def knn(data, target, test, n = 3, weights = "uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None):
    knn_estimator = KNeighborsClassifier(n, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
    knn_estimator.fit(data, target)
    result_arr = knn_estimator.predict(test)
    return result_arr

def decision_tree(data, target, parameters):
    parameters = {
        #'criterion': ['gini', 'entropy'],
        #'max_depth': [1, 2, 3, 4, 5, None],
        #'min_samples_split': [2, 3, 4, 5]
    # d_tree = tree.DecisionTreeClassifier(parameters)
    }
    pass


def general_training(best_model, data, target, test):
    best_model.fit(data,target)
    result_arr = best_model.predict(test)
    return result_arr

def train_xgb_model(params, x_train, y_train, rounds, early_stopping, n_folds, random_state, stratified = True,i=0, shuffle = True):

    # Model and hyperparameter selection

    ## grid search

    ## Model Training
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_folds=n_folds, random_state=random_state, shuffle=shuffle)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        d_train = xgb.DMatrix(x_train_cv, label=y_train_cv)
        d_valid = xgb.DMatrix(x_val_cv, label=y_val_cv)

        watchlist = [(d_train, 'train'), (d_valid, 'eval')]

        # declare your model
        xgb_model = xgb.train(params,
                              d_train,
                              rounds,  # number of rounds
                              watchlist,
                              early_stopping_rounds=early_stopping,
                              verbose_eval=50
                              )

        # predict train and validation set accuracy and get eval metrics
        scores_cv = xgb_model.predict(x_train_cv)
        scores_val = xgb_model.predict(x_val_cv)

        # evaluation
        #train_confusion_matrix = confusion_matrix(y_train_cv, np.around(scores_cv).astype(int))
        #val_confusion_matrix = confusion_matrix(y_val_cv, np.around(scores_val).astype(int))

        train_pc = accuracy_score(y_train_cv, scores_cv)
        train_pp = precision_score(y_train_cv, scores_cv)
        train_re = recall_score(y_train_cv, scores_cv)
        print('\n train-Accuracy: %.6f' % train_pc)
        print(' train-Precision: %.6f' % train_pp)
        print(' train-Recall: %.6f' % train_re)

        eval_pc = accuracy_score(y_val_cv,scores_val)
        eval_pp = precision_score(y_val_cv,scores_val)
        eval_re = recall_score(y_val_cv,scores_val)
        print('\n eval-Accuracy: %.6f' % eval_pc)
        print(' eval-Precision: %.6f' % eval_pp)
        print(' eval-Recall: %.6f' % eval_re)

        i = i + 1

    # return model for evaluation and prediction
    return xgb_model
