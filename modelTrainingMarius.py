import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold


def train_lgbm_model(x_train, y_train, params, n_folds, early_stopping, random_state, stratified = True, i=0, shuffle = True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    lgbm_model = LGBMClassifier(boosting_type=params['boosting_type'],
                                objective=params['objective'],
                                #n_jobs=params['n_jobs'],
                                n_estimators=params['n_estimators'],
                                silent=True,
                                num_leaves=params['num_leaves'],
                                max_depth = params['max_depth'],
                                #max_bin = params['max_bin'],
                                learning_rate=params['learning_rate'],
                                #subsample_for_bin = params['subsample_for_bin'],
                                #subsample = params['subsample'],
                                #colsample_bytree=params['colsample_bytree'],
                                #class_weight=params['class_weight'],
                                #subsample_freq = params['subsample_freq'],
                                #min_split_gain = params['min_split_gain'],
                                #min_child_weight = params['min_child_weight'],
                                #min_child_samples = params['min_child_samples'],
                                #reg_alpha=params['reg_alpha'],
                                #reg_lambda=params['reg_lambda'],
                                #random_state=params['random_state'],
                                #importance_type=params['importance_type'],
                                #scale_pos_weight = params['scale_pos_weight']
                                )

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        lgbm_model.fit(x_train_cv, y_train_cv, early_stopping_rounds=early_stopping, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

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

def train_xgb_model(params, fit_params,x_train, y_train, n_folds, random_state, stratified = True, i=0, shuffle = True):

    # Model and hyperparameter selection
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    xgb_model = XGBClassifier(**params)

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        xgb_model.fit(x_train_cv, y_train_cv, fit_params, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

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




