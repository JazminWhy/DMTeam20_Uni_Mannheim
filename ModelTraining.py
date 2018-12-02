from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from ModelEvaluation import *
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold

__author__ = "Marius Bock, Hendrik Roeder, Jasmin Weimueller"

######################################### DISCLAIMER ###################################################################

# This class features all methods used for training models.
# It as well incorporates the grid search function for finding the best model based on our custom profit function.

########################################################################################################################

def train_naive_bayes(params, x_train, y_train, n_folds, random_state, stratified = True, shuffle = True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    naive_bayes_model = GaussianNB(**params)
    i = 0

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


def train_nearest_centroid(params, x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    nearest_centroid_model = NearestCentroid(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        nearest_centroid_model.fit(x_train_cv, y_train_cv)

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


def train_knn(params,x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    knn_model = KNeighborsClassifier(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        knn_model.fit(x_train_cv, y_train_cv)

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


def train_decision_tree(params,x_train, y_train, n_folds, random_state, stratified = True, shuffle = True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    dt_tree_model = tree.DecisionTreeClassifier(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        dt_tree_model.fit(x_train_cv, y_train_cv)

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


def train_svm(params, x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    sv_model = svm.SVC(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        sv_model.fit(x_train_cv, y_train_cv )

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


def train_logistic(params,x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    log_model = LogisticRegression(**params)
    i = 0

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

        i = i + 1

    # return model for evaluation and prediction
    return log_model


#Complement NB, especially suited for imbalanced Datasets
def train_complement_naivebayes(params,x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    cnb_model = ComplementNB(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        cnb_model.fit(x_train_cv, y_train_cv )

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

        i = i + 1

    # return model for evaluation and prediction
    return cnb_model


def train_bernoulli_naivebayes(params,x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    bnb_model = BernoulliNB(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        bnb_model.fit(x_train_cv, y_train_cv )

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
        i = i + 1

    # return model for evaluation and prediction
    return bnb_model


def train_random_forest(params,x_train, y_train, n_folds, random_state, stratified=True, shuffle=True):
    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    rf_model = RandomForestClassifier(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        rf_model.fit(x_train_cv, y_train_cv )

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

        i = i + 1

    # return model for evaluation and prediction
    return rf_model


def train_xgb_model(x_train, y_train, params, n_folds, random_state, stratified=True, shuffle=True):

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

        # declare your model
        xgb_model.fit(x_train_cv, y_train_cv, verbose=False, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)], eval_metric='logloss')

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


def train_xgb_ensemble(x_train_b, y_train_b, x_train, y_train, x_test, y_test, params, n_folds, random_state, stratified=True, shuffle=True):

    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    ensemble_model = XGBClassifier(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train_b, y_train_b):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train_b.iloc[train_index], x_train_b.iloc[test_index]
        y_train_cv, y_val_cv = y_train_b.iloc[train_index], y_train_b.iloc[test_index]

        # declare your model
        ensemble_model.fit(x_train_cv, y_train_cv, verbose=False, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)], eval_metric='logloss')

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

    x_full = x_train.append(x_test)
    y_full = y_train.append(y_test)
    y_pred_full = ensemble_model.predict(x_full)
    y_pred_test = ensemble_model.predict(x_test)

    print("Whole dataset score:")
    print(profit_score_function(y_full, y_pred_full))
    print("Confusion")
    confusion_matrix_report(y_full, y_pred_full)
    print("Acc")
    print(accuracy_score(y_full, y_pred_full))
    print("Precision")
    print(precision_score(y_full, y_pred_full))
    print("Recall")
    print(recall_score(y_full, y_pred_full))
    print("F1")
    print(f1_score(y_full, y_pred_full))

    print("Test dataset score:")
    print(profit_score_function(y_test, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test, y_pred_test))
    print("Precision")
    print(precision_score(y_test, y_pred_test))
    print("Recall")
    print(recall_score(y_test, y_pred_test))
    print("F1")
    print(f1_score(y_test, y_pred_test))


def train_rf_ensemble(x_train_b, y_train_b, x_train, y_train, x_test, y_test, params, n_folds, random_state, stratified=True, shuffle=True):

    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    ensemble_model = RandomForestClassifier(**params)
    i = 0

    # Model Training
    for (train_index, test_index) in kf.split(x_train_b, y_train_b):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train_b.iloc[train_index], x_train_b.iloc[test_index]
        y_train_cv, y_val_cv = y_train_b.iloc[train_index], y_train_b.iloc[test_index]

        # declare your model
        ensemble_model.fit(x_train_cv, y_train_cv)

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

    x_full = x_train.append(x_test)
    y_full = y_train.append(y_test)
    y_pred_full = ensemble_model.predict(x_full)
    y_pred_test = ensemble_model.predict(x_test)

    print("Whole dataset score:")
    print(profit_score_function(y_full, y_pred_full))
    print("Confusion")
    confusion_matrix_report(y_full, y_pred_full)
    print("Acc")
    print(accuracy_score(y_full, y_pred_full))
    print("Precision")
    print(precision_score(y_full, y_pred_full))
    print("Recall")
    print(recall_score(y_full, y_pred_full))
    print("F1")
    print(f1_score(y_full, y_pred_full))

    print("Test dataset score:")
    print(profit_score_function(y_test, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test, y_pred_test))
    print("Precision")
    print(precision_score(y_test, y_pred_test))
    print("Recall")
    print(recall_score(y_test, y_pred_test))
    print("F1")
    print(f1_score(y_test, y_pred_test))


############################################## GRID SEARCH FUNCTION ####################################################


log = LogisticRegression()
rdf = RandomForestClassifier()
bnb = BernoulliNB()
mlp = MLPClassifier()
ncc = NearestCentroid()
knn = KNeighborsClassifier()
gnb = GaussianNB()
cnb = ComplementNB()
dt = tree.DecisionTreeClassifier()
xgb = XGBClassifier()
sv = svm.SVC()


ClassifierDict ={"LogisticRegression": log,
                 "DecisionTree": dt,
                 "RandomForest": rdf,
                 "KNN": knn,
                 "NearestCentroid":ncc,
                 "GaussianNB": gnb,
                 "BernoulliNB": bnb,
                 "ComplementNB": cnb,
                 "MLP": mlp,
                 "XGBoost": xgb,
                 "SupportVectorMachine": sv
                 }


def search_best_params_and_evaluate_general_model(classifier, X_full, y_full, X_train, y_train, X_test, y_test, parameter_dict, n_folds=5, fit_params=None):

    clf = ClassifierDict.get(classifier)
    best_model = grid_search_cost_model(model=clf,
                                        features=X_train,
                                        target=y_train,
                                        parameters=parameter_dict,
                                        folds=n_folds,
                                        fit_params=fit_params
                                        )

    best_params = best_model.get_params()
    y_pred_full = best_model.predict(X_full)
    y_pred_test = best_model.predict(X_test)

    print("Whole dataset score:")
    print(profit_score_function(y_full, y_pred_full))
    print("Confusion")
    confusion_matrix_report(y_full, y_pred_full)
    print("Acc")
    print(accuracy_score(y_full, y_pred_full))
    print("Precision")
    print(precision_score(y_full, y_pred_full))
    print("Recall")
    print(recall_score(y_full, y_pred_full))
    print("F1")
    print(f1_score(y_full, y_pred_full))

    print("Test dataset score:")
    print(profit_score_function(y_test, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test, y_pred_test))
    print("Precision")
    print(precision_score(y_test, y_pred_test))
    print("Recall")
    print(recall_score(y_test, y_pred_test))
    print("F1")
    print(f1_score(y_test, y_pred_test))

    return best_params


