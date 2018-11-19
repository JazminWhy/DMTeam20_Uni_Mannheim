# Supress unnecessary warnings so that presentation looks clean
import warnings

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

import pandas as pd
from modelTrainingMarius import *
import modelEvaluation
import xgboost as xgb
from modelEvaluation import *
from modelTrainingJasmin import *
### Data Loading #####

#Print all rows and columns. Dont hide any.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

bankingcalldata = pd.read_csv('C:/Users/hroed/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')
#C:\Users\jawei\PycharmProjects\DMTeam20_Uni_Mannheim\input
#### This is just for the model guys to train test their models #######

print('Full dataset shape: ')
print(bankingcalldata.shape)

from datetime import datetime
from sklearn.metrics import mean_absolute_error, accuracy_score, average_precision_score, recall_score, confusion_matrix
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import itertools

if bankingcalldata.isnull().values.any() == True:
    print('There are missing values in the dataset.')
else:
    print('There are no missing values in the dataset.')

columns = list(bankingcalldata.columns)

for column in columns:
    if bankingcalldata[column].isnull().values.any() == True:
        print('There are missing values in the column ' + column)

# Variable to hold the list of variables for an attribute in the train and test data
labels = []
to_be_encoded = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                 'previous','poutcome']

print('Encoding categorical columns..')

for i in bankingcalldata.columns.values:
    if bankingcalldata[i].dtype == object:
        lbl = preprocessing.LabelEncoder()
        bankingcalldata[i] = lbl.fit_transform(bankingcalldata[i])

for i in range(len(to_be_encoded)):
    labels.append(list(bankingcalldata[to_be_encoded[i]].unique()))

# One hot encode all categorical attributes
cats = []
encoded_data = bankingcalldata.drop('y', axis=1)

print('Checking datatypes..')
tmp = 0
for i in bankingcalldata.columns.values:
    if bankingcalldata[i].dtype == object:
        tmp = tmp + 1
if tmp == 0:
    print('All columns are encoded.')
else:
    print('Not all columns are encoded')

print('Finished.')

X_full = bankingcalldata.drop('y', axis=1)
y_full = bankingcalldata['y']
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.40, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_full.value_counts())

#### END ##############
"""
xgb_params_1 = {
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'max_depth': 5,
        #'silent': False,
        #'booster': 'gbtree',
        'min_child_weight': 1,
        #'max_delta_step':10,
        #'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        #'colsample_bylevel':0.8,
        'objective': 'binary:logistic',
        #'nthread': -1,
        #'scale_pos_weight': 1,
        #'base_score':1,
        #'random_state': 27,
        'eval_metric': 'logloss',
        #importance_type: 'gain',
        #'missing':np.nan,
        #'reg_alpha':10,
        #'reg_lambda': 10,
        #'num_class': 1,
    }

xgb_fit_params_1 = {
    #'verbose': True,
}

xgb_model_1 = modelTraining.train_xgb_model(xgb_params_1,
                                            xgb_fit_params_1,
                                            X_train,
                                            y_train,
                                            early_stopping=50,
                                            n_folds=10,
                                            random_state=123)
"""

knn = KNeighborsClassifier()
params_knn ={"n_neighbors": [2, 3],
             "algorithm": ['auto', 'ball_tree']} #, 'kd_tree', 'brute'
grid_search_model(model=knn, features=X_train, target=y_train, positive_label=1, parameters=params_knn, score="recall"
                  , fit_params=None, folds=5)
print("done")

best_model = grid_search_model(model=knn, features=X_train, target=y_train, positive_label=1, parameters=params_knn,
                               fit_params=None, score="recall", folds=5)
result_knn = general_training(best_model, data=X_train, target=y_train, test=X_test)
confusion_matrix_report(y_test, result_knn)

lgbm_params_1 = {
        'boosting_type': 'gbdt',
        'num_leaves': 1000,
        'max_depth': 10,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        #'subsample_for_bin': 1,
        'objective': 'binary',
        #'class_weight': 0,
        #'min_split_gain': 0.8,
        #'min_child_weight': 0.8,
        #'min_child_samples':0.8,
        #'subsample': 'binary:logistic',
        #'subsample_freq': -1,
        #'colsample_bytree': 1,
        #'reg_alpha':1,
        #'reg_lambda': 27,
        #'random_state': 'logloss',
        #'n_jobs': 'gain',
        #'silent':np.nan,
        #'importance_type':10,
    }

#lgbm_fit_params_1 = {
#    'early_stopping_rounds': 50,
#}

#
# xgb_model_1 = modelTraining.train_lgbm_model(X_train, y_train,
#                                              params= lgbm_params_1,
#                                              n_folds=10,
#                                              early_stopping=50,
#                                              random_state=123)
#
# xgb_model_1 = xgb.XGBClassifier()
# params_xgb ={ 'learning_rate':  [0.01,0.1],
#         'n_estimators': [100],#,1000],
#         'max_depth': [4]}#,5]}
#
# fit_params = {
#     #'early_stopping_rounds': 50,
#     #'verbose': 50
# }
# 
# grid_search(xgb_model_1, features=X_train, target=y_train, positive_label=1, parameters=params_xgb, score="kuchen",fit_params = fit_params, folds=2)

# print("start")
# result_knn = knn(data=X_train, target=y_train, test=X_test)
# print("here")
# print(result_knn)
# print("finish")
# confusion_matrix_report(y_test,result_knn)

#knn_grid = KNeighborsClassifier()
#params_knn ={"n_neighbors":[2,3,4,5],"algorithm":['auto', 'ball_tree']}#, 'kd_tree', 'brute']}
#grid_search_f1(model=knn_grid, features=X_train, target=y_train, positive_label=1, parameters=params_knn)
print("done")

#best_model = grid_search_model(model=knn_grid, features=X_train, target=y_train, positive_label=1, parameters=params_knn, fit_params=None, score="recall", folds=2)
#result_knn = general_training(best_model, data=X_train, target=y_train, test=X_test)
#confusion_matrix_report(y_test,result_knn)

#best f1 is 0.6102194941535 with params {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 35, 'num_leaves': 100, 'objective': 'binary'}

#lgbm_model = LGBMClassifier()
#params_lgbm ={"objective": ["binary"],
#              "n_estimators": [32, 33, 34, 35, 36, 37, 38],
#              "num_leaves": [97, 98, 99, 100, 101, 102, 103],
#              "max_depth": [-1, 999999999],
#              "learning_rate": [0.05, 0.1, 0.15, 0.2]}

#best_lgbm = grid_search_model(model=lgbm_model, features=X_train, target=y_train, positive_label=1,
#                              parameters=params_lgbm, fit_params=None, score="f1", folds=5)
#result_lgbm = general_training(best_model=best_lgbm, data=X_train, target=y_train, test=X_test)
#confusion_matrix_report(y_test, result_lgbm)


xgb_model = XGBClassifier()
params2 = {
       'learning_rate': [0.1],
       'n_estimators': [1000],
       'max_depth': [5],
       'min_child_weight': [1],
       'gamma': [0],
       'subsample': [0.8],
       'colsample_bytree': [0.8],
       'objective': ['binary:logistic'],
       'nthread': [-1],
       'scale_pos_weight': [1],
       'seed': [27],
       'eval_metric': ['logloss'],
       'num_class': [1],
   }
params_xgb ={"Eta": [0.01],
             "Gamma": [0],
             "Max_depth": [3, 7, 12, 15],
             "Min_child_weight": [1, 3, 7, 10],
             "Subsample": [0.8],
             "Colsample_bytree": [0.8],
             "Lambda": [0.01],
             "alpha": [0],
             "learning_rate": [0.1],
             "n_estimators": [140],
             "objective": ["binary:logistic"],
             "nthread": [4],
             "scale_pos_weight": [1],
             "seed": [27]
             }
print('start grid')
#best_xgb = grid_search_model(model=xgb_model, features=X_train, target=y_train, positive_label=1,
#                              parameters=params2, fit_params=None, score="f1", folds=5)
print('Grid done')
#result_xgb = general_training(best_model=best_xgb, data=X_train, target=y_train, test=X_test)
#confusion_matrix_report(y_test, result_xgb)

