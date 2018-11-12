#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import classifiers & modules


# In[2]:


# get Features - method 


# In[1]:


# CLASSIFIER 1
#Import the library

import sklearn as sk
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd

bankingcalldata = pd.read_csv('datasets/BankingCallData/bank-additional/bank-additional-full.csv', sep=';')
y_full = bankingcalldata['y']
x_full = bankingcalldata.drop('y', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.40, random_state=42)

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

def ridge_model(alpha, X, y):
    
    ridge_model = RidgeClassifier(alpha, fit_intercept= True, normalize=True)
    ridge_model.fit(X, y)
    return ridge_model


def run_classifier1_replace_name(n_folds, random_state, x_train, y_train, alpha=0.1, i=0, shuffle=True):
    
    # Model and hyperparameter selection
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    # Model Training  
    for (train_index, test_index) in skf.split(x_train, y_train):
        # cross-validation randomly splits train data into train and validation data
        print('\n Fold %d' % (i + 1))
    
        x_train_cv, x_val_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[test_index]
 
        # declare your model
        model = ridge_model(alpha, x_train_cv, y_train_cv)
    
        # predict train and validation set accuracy and get eval metrics
        scores_cv = model.predict(x_train_cv)
        scores_val = model.predict(x_val_cv)
        
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
    
        # predict the test data and add it to the other predictions
        test_pred = model.predict(x_test)
        train_pred = model.predict(x_train)
        full_pred = model.predict(x_full)
        
        if i > 0:
            temp_sum_train = sum_train_pred + train_pred
            temp_sum_test = sum_test_pred + test_pred
            temp_sum_full = sum_full_pred + full_pred
        else:
            temp_sum_train = train_pred
            temp_sum_test = test_pred
            temp_sum_full = full_pred
        sum_train_pred = temp_sum_train
        sum_test_pred = temp_sum_test
        sum_full_pred = temp_sum_full

        i = i+1
        # return output for evaluation
    mean_fold_results(sum_train_pred, sum_test_pred, sum_full_pred, y_train, y_test, n_folds)


def mean_fold_results(sum_train_pred, sum_test_pred, sum_full_pred, y_train, y_test, n_folds):
    # divide predictions and CV-sum by number of folds to get mean of all folds
    final_train_pred_ridge = sum_train_pred / (n_folds)
    final_test_pred_ridge = sum_test_pred / (n_folds)
    final_full_pred_ridge = sum_full_pred / (n_folds)
    
    train_confusion_matrix_ridge = confusion_matrix(y_train, np.around(final_train_pred_ridge).astype(int))
    test_confusion_matrix_ridge = confusion_matrix(y_test, np.around(final_test_pred_ridge).astype(int))
    full_confusion_matrix_ridge = confusion_matrix(y_full, np.around(final_full_pred_ridge).astype(int))
    
    final_train_accuracy_ridge = accuracy(train_confusion_matrix_ridge)
    final_train_precision_ridge = precision(train_confusion_matrix_ridge)
    final_train_recall_ridge = recall(train_confusion_matrix_ridge)
    
    final_test_accuracy_ridge = accuracy(test_confusion_matrix_ridge)
    final_test_precision_ridge = precision(test_confusion_matrix_ridge)
    final_test_recall_ridge = recall(test_confusion_matrix_ridge)
    
    final_full_accuracy_ridge = accuracy(full_confusion_matrix_ridge)
    final_full_precision_ridge = precision(full_confusion_matrix_ridge)
    final_full_recall_ridge = recall(full_confusion_matrix_ridge)
    
    print('\n Average Ridge full-Accuracy: %.6f' % final_full_accuracy_ridge)
    print(' Average Ridge full-Precision: %.6f' % final_full_precision_ridge)
    print(' Average Ridge full-Recall: %.6f' % final_full_recall_ridge)



# In[4]:




