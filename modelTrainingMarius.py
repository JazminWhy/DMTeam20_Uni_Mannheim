import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


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

        # declare your model
        xgb_model.fit(x_train_cv, y_train_cv, early_stopping_rounds=50, eval_set=[(x_train_cv, y_train_cv), (x_val_cv, y_val_cv)])

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


def train_ensemble(models, y_train, ensemble_algorithm, params, fit_params, n_folds, random_state, stratified = True, shuffle = True):

    # stratified yes/ no
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    ensemble_model = ensemble_algorithm(**params)
    i = 0
    j = 0

    x_train = pd.Dataframe()

    for model in models:
        x_train['model_'&j] = model.predict_proba(y_train)
        j = j + 1

    # Model Training
    for (train_index, test_index) in kf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))

        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # declare your model
        ensemble_model.fit(x_train_cv, y_train_cv, fit_params)

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



