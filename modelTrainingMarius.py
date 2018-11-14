import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

def accuracy(array):
    TN = array[0, 0]
    TP = array[1, 1]
    FN = array[1, 0]
    FP = array[0, 1]

    return ((TP + TN) / (TN + TP + FN + FP))


def precision(array):
    TN = array[0, 0]
    TP = array[1, 1]
    FN = array[1, 0]
    FP = array[0, 1]

    return (TP / (TP + FP))


def recall(array):
    TN = array[0, 0]
    TP = array[1, 1]
    FN = array[1, 0]
    FP = array[0, 1]

    return (TP / (TP + FN))

def train_xgb_model(params, x_train, y_train, rounds, early_stopping, n_folds, random_state, i=0, shuffle = True):

    # Model and hyperparameter selection
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    # Model Training
    for (train_index, test_index) in skf.split(x_train, y_train):
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
        train_confusion_matrix = confusion_matrix(y_train_cv, np.around(scores_cv).astype(int))
        val_confusion_matrix = confusion_matrix(y_val_cv, np.around(scores_val).astype(int))

        train_pc = accuracy(train_confusion_matrix)
        train_pp = precision(train_confusion_matrix)
        train_re = recall(train_confusion_matrix)
        print('\n train-Accuracy: %.6f' % train_pc)
        print(' train-Precision: %.6f' % train_pp)
        print(' train-Recall: %.6f' % train_re)

        eval_pc = accuracy(val_confusion_matrix)
        eval_pp = precision(val_confusion_matrix)
        eval_re = recall(val_confusion_matrix)
        print('\n eval-Accuracy: %.6f' % eval_pc)
        print(' eval-Precision: %.6f' % eval_pp)
        print(' eval-Recall: %.6f' % eval_re)

        i = i + 1

    # return model for evaluation and prediction
    return xgb_model




