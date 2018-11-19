import xgboost as xgb
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.datasets import load_iris
from skrules import SkopeRules
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn import svm
# implements NaiveBayes, KNN, NearestCentroid, DecisionTree, GeneralModel

def train_naive_bayes(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    #naive_bayes = GaussianNB()
    #naive_bayes.fit(golf_encoded, golf['Play'])
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    naive_bayes_model = GaussianNB(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        naive_bayes_model.fit(x_train_cv, y_train_cv)

        # predict train and validation set accuracy and get eval metrics
        scores_cv = naive_bayes_model.predict(x_train_cv)
        scores_val = naive_bayes_model.predict(x_val_cv)

        # training evaluation

        train_pc = accuracy_score(y_train_cv, scores_cv)
        train_pp = precision_score(y_train_cv, scores_cv)
        train_re = recall_score(y_train_cv, scores_cv)
        print('\n train-Accuracy: %.6f' % train_pc)
        print(' train-Precision: %.6f' % train_pp)
        print(' train-Recall: %.6f' % train_re)

        eval_pc = accuracy_score(y_val_cv, scores_val)
        eval_pp = precision_score(y_val_cv, scores_val)
        eval_re = recall_score(y_val_cv, scores_val)
        print('\n eval-Accuracy: %.6f' % eval_pc)
        print(' eval-Precision: %.6f' % eval_pp)
        print(' eval-Recall: %.6f' % eval_re)

        i = i + 1

    # return model for evaluation and prediction
    return naive_bayes_model

def train_nearest_centroid(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):

    # nearest_centroid_estimator = NearestCentroid()
    # nearest_centroid_estimator.fit....
    # result_arr_knn = knn_estimator.predict(test)
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    nearest_centroid_model = NearestCentroid(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        nearest_centroid_model.fit(x_train_cv, y_train_cv)#], fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = nearest_centroid_model.predict(x_train_cv)
        scores_val = nearest_centroid_model.predict(x_val_cv)

        # training evaluation

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
    return nearest_centroid_model


def train_knn(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
                #(data, target, test, n = 3, weights = "uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None):
    # knn_estimator = KNeighborsClassifier(n, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
    # knn_estimator.fit(data, target)
    # result_arr = knn_estimator.predict(test)
    # return result_arr
    #
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    knn_model = KNeighborsClassifier(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        knn_model.fit(x_train_cv, y_train_cv)#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = knn_model.predict(x_train_cv)
        scores_val = knn_model.predict(x_val_cv)

        # training evaluation

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
    return knn_model


def train_decision_tree(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    dt_tree_model = tree.DecisionTreeClassifier(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        dt_tree_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = dt_tree_model.predict(x_train_cv)
        scores_val = dt_tree_model.predict(x_val_cv)

        # training evaluation

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
    return dt_tree_model

def train_general_model_results(best_model, x_train, y_train, x_test):
    best_model.fit(x_train,y_train)
    result_arr = best_model.predict(x_test)
    return result_arr

def predict_general_model_results(best_model, x_test):
    result_arr = best_model.predict(x_test)
    return result_arr

def train_general_model(best_model, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        best_model.fit(x_train_cv, y_train_cv)#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = best_model.predict(x_train_cv)
        scores_val = best_model.predict(x_val_cv)

        # training evaluation

        train_pc = accuracy_score(y_train_cv, scores_cv)
        train_pp = precision_score(y_train_cv, scores_cv)
        train_re = recall_score(y_train_cv, scores_cv)
        print('\n train-Accuracy: %.6f' % train_pc)
        print(' train-Precision: %.6f' % train_pp)
        print(' train-Recall: %.6f' % train_re)

        eval_pc = accuracy_score(y_val_cv, scores_val)
        eval_pp = precision_score(y_val_cv, scores_val)
        eval_re = recall_score(y_val_cv, scores_val)
        print('\n eval-Accuracy: %.6f' % eval_pc)
        print(' eval-Precision: %.6f' % eval_pp)
        print(' eval-Recall: %.6f' % eval_re)

        i = i + 1

    # return model for evaluation and prediction
    return best_model

def skope_rules(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    clf = SkopeRules(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model

        clf.fit(x_train_cv, y_train_cv)#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = clf.score_top_rules(x_train_cv)
        scores_val = clf.score_top_rules(x_val_cv)

        # training evaluation

        # train_pc = accuracy_score(y_train_cv, scores_cv)
        # train_pp = precision_score(y_train_cv, scores_cv)
        # train_re = recall_score(y_train_cv, scores_cv)
        # print('\n train-Accuracy: %.6f' % train_pc)
        # print(' train-Precision: %.6f' % train_pp)
        # print(' train-Recall: %.6f' % train_re)
        #
        # eval_pc = accuracy_score(y_val_cv,scores_val)
        # eval_pp = precision_score(y_val_cv,scores_val)
        # eval_re = recall_score(y_val_cv,scores_val)
        # print('\n eval-Accuracy: %.6f' % eval_pc)
        # print(' eval-Precision: %.6f' % eval_pp)
        # print(' eval-Recall: %.6f' % eval_re)
        precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve')
        plt.show()

        i = i + 1

    # return model for evaluation and prediction
    return clf


def train_svm(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    sv_model = svm.SVC(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        sv_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = sv_model.predict(x_train_cv)
        scores_val = sv_model.predict(x_val_cv)

        # training evaluation

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
    return sv_model