from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from skrules import SkopeRules
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
# implements NaiveBayes, KNN, NearestCentroid, DecisionTree, GeneralModel
from ModelEvaluation import *
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU



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

    #sv_model = svm.OneClassSVM(**params)
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

        # precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve')
        # plt.show()
        i = i + 1

    # return model for evaluation and prediction
    return sv_model


def train_multilayerperceptron(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    #sv_model = svm.OneClassSVM(**params)
    mlp_model = MLPClassifier(**params)
    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        mlp_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = mlp_model.predict(x_train_cv)
        scores_val = mlp_model.predict(x_val_cv)

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

        # precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve')
        # plt.show()
        i = i + 1

    # return model for evaluation and prediction
    return mlp_model

def train_logistic(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    #sv_model = svm.OneClassSVM(**params)
    log_model = LogisticRegression(**params)
    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        log_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = log_model.predict(x_train_cv)
        scores_val = log_model.predict(x_val_cv)

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

        # precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve')
        # plt.show()
        i = i + 1

    # return model for evaluation and prediction
    return log_model

#Complement NB, especially suited for imbalanced Datasets
def train_complement_naiveBayes(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    #sv_model = svm.OneClassSVM(**params)
    cnb_model = ComplementNB(**params)
    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        cnb_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = cnb_model.predict(x_train_cv)
        scores_val = cnb_model.predict(x_val_cv)

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

        # precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve')
        # plt.show()
        i = i + 1

    # return model for evaluation and prediction
    return cnb_model


def train_Bernoulli_NaiveBayes(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    #sv_model = svm.OneClassSVM(**params)
    bnb_model = BernoulliNB(**params)
    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        bnb_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = bnb_model.predict(x_train_cv)
        scores_val = bnb_model.predict(x_val_cv)

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

        # precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve')
        # plt.show()
        i = i + 1

    # return model for evaluation and prediction
    return bnb_model

def train_Random_Forests(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    #sv_model = svm.OneClassSVM(**params)
    rf_model = RandomForestClassifier(**params)
    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        rf_model.fit(x_train_cv, y_train_cv )#, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = rf_model.predict(x_train_cv)
        scores_val = rf_model.predict(x_val_cv)

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

        # precision, recall, _ = precision_recall_curve(y_train_cv, scores_cv)
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve')
        # plt.show()
        i = i + 1

    # return model for evaluation and prediction
    return rf_model


def declare_nn_model(input_dim):

    # create model
    nn_model = Sequential()
    nn_model.add(Dense(60, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    nn_model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    nn_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # compile model
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn_model


def train_nn_model(x_train, y_train, n_folds, epochs, random_state, stratified = True, shuffle = True):


    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        nn_model = declare_nn_model(x_train_cv.shape[1])

        # fit model
        nn_model.fit(np.array(x_train_cv), np.array(y_train_cv), epochs=epochs, verbose= 2,
                     validation_data=(np.array(x_val_cv), np.array(y_val_cv)))

        # predict train and validation set accuracy and get eval metrics
        scores_cv = nn_model.predict(x_train_cv)
        scores_val = nn_model.predict(x_val_cv)

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
    return nn_model


def declare_lgbm_model(params):

    # declare model
    lgbm_model = LGBMClassifier(boosting_type=params['boosting_type'],
                                objective=params['objective'],
                                # n_jobs=params['n_jobs'],
                                n_estimators=params['n_estimators'],
                                silent=True,
                                num_leaves=params['num_leaves'],
                                max_depth=params['max_depth'],
                                # max_bin = params['max_bin'],
                                learning_rate=params['learning_rate'],
                                # subsample_for_bin = params['subsample_for_bin'],
                                # subsample = params['subsample'],
                                # colsample_bytree=params['colsample_bytree'],
                                # class_weight=params['class_weight'],
                                # subsample_freq = params['subsample_freq'],
                                # min_split_gain = params['min_split_gain'],
                                # min_child_weight = params['min_child_weight'],
                                # min_child_samples = params['min_child_samples'],
                                # reg_alpha=params['reg_alpha'],
                                # reg_lambda=params['reg_lambda'],
                                # random_state=params['random_state'],
                                # importance_type=params['importance_type'],
                                # scale_pos_weight = params['scale_pos_weight']
                                )

    return lgbm_model


def train_lgbm_model(x_train, y_train, params, n_folds, early_stopping, random_state, stratified = True, shuffle = True):

    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    lgbm_model = declare_lgbm_model(params)

    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        lgbm_model.fit(x_train_cv, y_train_cv, early_stopping_rounds=early_stopping, verbose = True, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)], eval_metric='logloss')

        # predict train and validation set accuracy and get eval metrics
        scores_cv = lgbm_model.predict(x_train_cv)
        scores_val = lgbm_model.predict(x_val_cv)

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
    return lgbm_model

def train_xgb_model(x_train, y_train, params, n_folds, random_state, stratified = True, shuffle = True):

    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    xgb_model = XGBClassifier(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        import xgboost


        # declare your model
        xgb_model.fit(x_train_cv, y_train_cv, early_stopping_rounds=50, verbose = 50, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

        # predict train and validation set accuracy and get eval metrics
        scores_cv = xgb_model.predict(x_train_cv)
        scores_val = xgb_model.predict(x_val_cv)

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
    return xgb_model


def train_xgb_ensemble(models, x_train ,y_train, params, n_folds, random_state, stratified = True, shuffle = True):

    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    ensemble_model = XGBClassifier(**params)
    i = 0
    j = 0

    x_train_probas = pd.DataFrame(index=None)

    print(y_train.shape)
    for model in models:
        x_train_probas[j] = model.predict_proba(x_train).tolist()
        print(x_train)
        j = j + 1

    # Model Training
    for (train_index, test_index) in kf.split(x_train_probas, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        ensemble_model.fit(x_train_cv, y_train_cv, early_stopping_rounds=50, verbose = True, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)], eval_metric='logloss')

        # predict train and validation set accuracy and get eval metrics
        scores_cv = ensemble_model.predict(x_train_cv)
        scores_val = ensemble_model.predict(x_val_cv)

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


############################################## GRID SEARCH FUNCTION ####################################################


log = LogisticRegression()
rdf = RandomForestClassifier()
bnb = BernoulliNB()
mlp = MLPClassifier()
#rules = skope_rules()
ncc = NearestCentroid()
knn = KNeighborsClassifier()
gnb = GaussianNB()
cnb = ComplementNB()
dt = tree.DecisionTreeClassifier()
xgb = XGBClassifier()
lgbm = LGBMClassifier()

ClassifierDict ={"LogisticRegression": log,
                 "DecisionTree":dt,
                 "RandomForest":rdf,
                 "KNN": knn,
                 "NearestCentroid":ncc,
                 "GaussianNB": gnb,
                 "BernoulliNB":bnb,
                 "ComplementNB":cnb,
                 "MLP":mlp,
                 #"rules":rules
                 "XGBoost":xgb
                 }

def search_best_params_and_evaluate_general_model(classifier, X_train, y_train, X_test, y_test, parameter_dict, n_folds=5, fit_params=None):

    clf = ClassifierDict.get(classifier)
    best_model = grid_search_cost_model(model=clf,
                                        features=X_train,
                                        target=y_train,
                                        parameters=parameter_dict,
                                        folds=n_folds,
                                        fit_params=fit_params
                                        )

    best_params = best_model.get_params()

    # predict train and validation set accuracy and get eval metrics
    results = predict_general_model_results(best_model, X_test)

    print("Profits")
    print(profit_score_function(y_test,results))
    print("Confusion")
    confusion_matrix_report(y_test,results)
    print("Acc")
    print(accuracy_score(y_test,results))
    print("Precision")
    print(precision_score(y_test,results))
    print("Recall")
    print(recall_score(y_test,results))
    print("F1")
    print(f1_score(y_test,results))

    return best_params


