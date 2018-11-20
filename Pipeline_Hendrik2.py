# Supress unnecessary warnings so that presentation looks clean
import random
import warnings

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

import pandas as pd
import modelTrainingMarius as modelTraining
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
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.20, random_state=42, stratify=y_full)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_full.value_counts())
print(y_train.value_counts())
balance = True
if balance==True:
    #### END ####################### BALANCING data
    y_train = pd.DataFrame(data=y_train)
    y_train_balance = y_train[y_train==1]
    y_train_0 = y_train[y_train==0]
    train_full_balance= pd.DataFrame(data=X_train)
    train_full_balance = train_full_balance.assign(y = y_train["y"])
    train_full_balance.head()
    train_x_balance = train_full_balance[train_full_balance["y"]== 1]
    print(train_x_balance.shape)
    train_x_balance_0 = train_full_balance[train_full_balance["y"]== 0]
    print(train_x_balance_0.shape)
    print("Y_1 count ")
    count_pos = train_x_balance.count(axis="rows")["age"]
    train_x_balance_0_sample=train_x_balance_0.sample(n=count_pos, replace=False, random_state=42) #random.sample(train_x_balance_0, count_pos)
    print(train_x_balance_0_sample.shape)
    train_full_balance = train_x_balance.append(train_x_balance_0_sample)
    print(train_full_balance.shape)
    X_train = train_full_balance.drop("y", axis=1)
    print(X_train.shape)
    y_train = train_full_balance["y"]
    print(y_train.shape)
################################
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
#
# knn_grid = KNeighborsClassifier()
# params_knn ={"n_neighbors":[2,3,4,5],"algorithm":['auto', 'ball_tree']}#, 'kd_tree', 'brute']}
# grid_search_f1(model=knn_grid, features=X_train, target=y_train, positive_label=1, parameters=params_knn)
# print("done")


################ GRID SEARCH NEAREST CENTROID
# nc_grid = NearestCentroid()
# params_cnn ={"metric":['euclidean', 'manhattan', 'minkowski', 'chebyshev', 'seuclidean', 'mahalanobis']}#, 'kd_tree', 'brute']}
# best_model = grid_search_model(model=nc_grid, features=X_train, target=y_train, positive_label=1, parameters=params_cnn, fit_params=None, score="roc_auc", folds=10)
# best_cnn_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_cnn = predict_general_model_results(best_cnn_model,x_test=X_test)
# confusion_matrix_report(y_test,result_cnn)
# print(accuracy_score(y_test,result_cnn))
# print(precision_score(y_test,result_cnn))
# print(recall_score(y_test,result_cnn))

################ GRID SEARCH KNN
#knn_grid = KNeighborsClassifier()
#params_knn = {"n_neighbors":[2,3,4,5],"algorithm":['auto', 'ball_tree','kd_tree', 'brute'], "metric":['euclidean', 'manhattan', 'minkowski', 'chebyshev']}
#best_model = grid_search_model(model=knn_grid, features=X_train, target=y_train, positive_label=1, parameters=params_knn, fit_params=None, score="precision", folds=10)
#best_knn_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
#result_knn = predict_general_model_results(best_knn_model,x_test=X_test)
#confusion_matrix_report(y_test,result_knn)
#print(accuracy_score(y_test,result_knn))
#print(precision_score(y_test,result_knn))
#print(recall_score(y_test,result_knn))

## test KNN
#params_knn = {'n_neighbors':3, 'weights' : "uniform", 'algorithm':"auto", 'leaf_size':30, 'p':2, 'metric':"minkowski", 'metric_params':None, 'n_jobs':None}
#knn_model = train_knn(params_knn, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

## test Decision_Trees
#params_dt = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_decrease':0.0, 'min_impurity_split':None, 'class_weight':None, 'presort':False}
#dt_model = train_decision_tree(params_dt, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

################ GRID SEARCH Dtree
dtree_grid = tree.DecisionTreeClassifier()
params_dtree = {'criterion':['gini', 'entropy'],
                'splitter':['best'],
                'max_depth':[5, 8, 15, 25, None],
                'min_samples_split':[2, 10, 100],
                'min_samples_leaf':[10, 20],
                'min_weight_fraction_leaf':[0.0, 0.5],
                #'max_features':None,
                #'random_state':None,
                #'max_leaf_nodes':None,
                'min_impurity_decrease':[0.0, 0.01],
                #'min_impurity_split':None,
                #'class_weight':None,
                #'presort':False
                }
best_dtree = grid_search_model(model=dtree_grid, features=X_train, target=y_train, positive_label=1,
                               parameters=params_dtree, fit_params=None, score="roc_auc", folds=10)
best_dtree_model = train_general_model(best_dtree, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None,
                                     random_state=123, stratified=True, i=0, shuffle=True)
result_dtree = predict_general_model_results(best_dtree_model, x_test=X_test)
confusion_matrix_report(y_test,result_dtree)
#print(accuracy_score(y_test,result_dtree))
print(precision_score(y_test,result_dtree))
print(recall_score(y_test,result_dtree))
print(f1_score(y_test,result_dtree))

#best roc_auc is 0.9357937805579077 with params {'criterion': 'gini', 'max_depth': 8, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 10, 'min_samples_split': 100, 'min_weight_fraction_leaf': 0.0, 'splitter': 'best'}
#        _Prediction_
#            0      1
#      0  7054    256
#      1   442    486
# 0.6549865229110512
# 0.5237068965517241
# 0.5820359281437126
## test NaiveBayes
# params_nb = {'priors': None, 'var_smoothing':1e-10}
# naive_bayes_model = train_naive_bayes(params_nb, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

## Test Nearest Centroid
#params_nearest_centroid = {'metric':'manhattan'}
#nearest_cetroid_model = train_nearest_centroid(params_nearest_centroid, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)


## test rule based classifier
# X_train_rules = X_train.drop('nr.employed', axis=1)
# X_train_rules = X_train_rules.drop('emp.var.rate', axis=1)
# X_train_rules = X_train_rules.drop('cons.conf.idx', axis=1)
# X_train_rules = X_train_rules.drop('cons.price.idx', axis=1)
#
# params_rule = {'max_depth_duplication':2,
#                  'n_estimators':30,
#                  'precision_min':0.4,
#                  'recall_min':0.05,
#                  'feature_names': list(X_train_rules.columns.values)}
# rules = skope_rules(params_rule, fit_params=None, x_train=X_train_rules, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
