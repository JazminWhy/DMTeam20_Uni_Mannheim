# Supress unnecessary warnings so that presentation looks clean
import warnings

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

import pandas as pd
import modelTrainingMarius as modelTraining
from modelEvaluation import *
from modelTrainingJasmin import *
from dataPreProcessing_Soumya import *
from sklearn.metrics import mean_absolute_error, accuracy_score, average_precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
### Data Loading #####

#Print all rows and columns. Dont hide any.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

bankingcalldata = pd.read_csv('C:/Users/hroed/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')
#C:\Users\jawei\PycharmProjects\DMTeam20_Uni_Mannheim\input
#### This is just for the model guys to train test their models #######

# Check missing values
check_missing_values(bankingcalldata)

print('Full dataset shape: ')
print(bankingcalldata.shape)

X_full = bankingcalldata.drop('y', axis=1)
y_full = bankingcalldata['y']
y_full.replace(('yes', 'no'), (1, 0), inplace=True)

cols = X_full.columns
num_cols = X_full._get_numeric_data().columns
columns_to_onehot = list(set(cols) - set(num_cols))
X_preprocessed = data_preprocessing(data_set=X_full,
                                                             columns_to_drop=[],
                                                             columns_to_onehot=columns_to_onehot,
                                                             columns_to_dummy=[],
                                                             columns_to_label=[],
                                                             normalise=True)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_full, test_size=0.40, random_state=42, stratify=y_full)
# Balancing
X_train, y_train = data_balancing(X_train=X_train, y_train=y_train)

print(y_test.value_counts())

print('shapes')
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#print(X_train.head())
print(y_full.value_counts())
print(y_train.value_counts())

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

## grid search with SVM

# params_svm_grid = {'kernel':['sigmoid','linear','poly','rbf'], 'class_weight': [None,'balanced']}
# svm = svm.SVC()
# best_model = grid_search_model(model=svm, features=X_train, target=y_train, positive_label=1, parameters=params_svm_grid, fit_params=None, score="precision", folds=5)
# best_svm_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_svm = predict_general_model_results(best_svm_model,x_test=X_test)
# confusion_matrix_report(y_test,result_svm)
# print(accuracy_score(y_test,result_svm))
# print(precision_score(y_test,result_svm))
# print(recall_score(y_test,result_svm))

## test SVM
# params_svm = {'C':1.0, 'cache_size':100, 'class_weight':'balanced', 'coef0':0.0,
#     'decision_function_shape':'ovr', 'degree':3, 'gamma':'scale', 'kernel':'linear',
#     'max_iter':-1, 'probability':False, 'random_state':123, 'shrinking':True,
#     'tol':0.001, 'verbose':False}
#
# params_One_svm = { 'cache_size':200, 'coef0':0.0,
#      'degree':3, 'gamma':'scale', 'kernel':'rbf',
#     'max_iter':-1,  'random_state':123, 'shrinking':True,
#     'tol':0.001, 'verbose':True}
# svm_model = train_svm(params_svm, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
# result_svm= predict_general_model_results(svm_model,x_test=X_test)
# confusion_matrix_report
# (y_test,result_svm)
# print(accuracy_score(y_test,result_svm))
# print(precision_score(y_test,result_svm))
# print(recall_score(y_test,result_svm))



####### grid search MLP
# params_mlp_grid = {'hidden_layer_sizes':[(100, ), ( 200, )], 'activation': ['logistic','relu','tanh'], 'solver':['sgd','adam','lbfgs'], 'learning_rate_init':[0.001, 0.0005]}
# mlp = MLPClassifier()
# best_model = grid_search_model(model=mlp, features=X_train, target=y_train, positive_label=1, parameters=params_mlp_grid, fit_params=None, score="precision", folds=5)
# best_mlp_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_mlp = predict_general_model_results(best_mlp_model,x_test=X_test)
# confusion_matrix_report(y_test,result_mlp)
# print(accuracy_score(y_test,result_mlp))
# print(precision_score(y_test,result_mlp))
# print(recall_score(y_test,result_mlp))

## test MLP
# params_mlp = {'hidden_layer_sizes':(100, ), 'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto', 'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True, 'random_state':None, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9, 'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08, 'n_iter_no_change':10}
# mlp_model = train_multilayerperceptron(params_mlp, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
# result_mlp= predict_general_model_results(mlp_model,x_test=X_test)
# confusion_matrix_report(y_test,result_mlp)
# print(accuracy_score(y_test,result_mlp))
# print(precision_score(y_test,result_mlp))
# print(recall_score(y_test,result_mlp))


## logistic regression grid searcb
# params_log_grid = {'penalty':['l2'], 'class_weight': [None,'balanced'], 'solver':['sag','newton-cg','lbfgs']}
# log = LogisticRegression()
# best_model = grid_search_model(model=log, features=X_train, target=y_train, positive_label=1, parameters=params_log_grid, fit_params=None, score="precision", folds=5)
# best_log_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_log = predict_general_model_results(best_log_model,x_test=X_test)
# confusion_matrix_report(y_test,result_log)
# print(accuracy_score(y_test,result_log))
# print(precision_score(y_test,result_log))
# print(recall_score(y_test,result_log))

## test logistic
# params_log = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':1.0, 'fit_intercept':True, 'intercept_scaling':1, 'class_weight':None, 'random_state':None, 'solver':'warn', 'max_iter':100, 'multi_class':'warn', 'verbose':0, 'warm_start':False, 'n_jobs':None}
# log_model = train_logistic(params_log, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
# result_log= predict_general_model_results(log_model,x_test=X_test)
# confusion_matrix_report(y_test,result_log)
# print(accuracy_score(y_test,result_log))
# print(precision_score(y_test,result_log))
# print(recall_score(y_test,result_log))

################ GRID SEARCH KNN
knn_grid = KNeighborsClassifier()
params_knn = {"n_neighbors":[16],
#              "algorithm":['auto', 'ball_tree','kd_tree', 'brute'],
 #             "metric":['euclidean', 'manhattan', 'minkowski', 'chebyshev']
              "p": [2]
              }
"""
best_model = grid_search_model(model=knn_grid, features=X_train, target=y_train, positive_label=1, parameters=params_knn, fit_params=None, score="f1", folds=2)
best_knn_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
result_knn = predict_general_model_results(best_knn_model,x_test=X_test)
confusion_matrix_report(y_test,result_knn)
print(accuracy_score(y_test,result_knn))
print(precision_score(y_test,result_knn))
print(recall_score(y_test,result_knn))
print(f1_score(y_test,result_knn))
"""
################ GRID SEARCH DTREE

dtree_grid = tree.DecisionTreeClassifier()
params_dtree = {'criterion':['gini', 'entropy'],
                'splitter':['best'],
                'max_depth':[2, 3],
                'min_samples_split':[2, 3, 4, 5],
                'min_samples_leaf':[18],
                'min_weight_fraction_leaf':[0.0],
                #'max_features':None,
                #'random_state':None,
                #'max_leaf_nodes':None,
                'min_impurity_decrease':[0.0],
                #'min_impurity_split':None,
                #'class_weight':None,
                #'presort':False
                }

best_model = grid_search_cost_model_balanced(model=dtree_grid, features=X_train, target=y_train, parameters=params_dtree, folds=10, fit_params=None)
#best_model = grid_search_model(model=dtree_grid, features=X_train, target=y_train, positive_label=1, parameters=params_dtree, fit_params=None, score="f1", folds=2)
best_dtree_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
result_dtree = predict_general_model_results(best_dtree_model,x_test=X_test)
confusion_matrix_report(y_test,result_dtree)
print(accuracy_score(y_test,result_dtree))
print(precision_score(y_test,result_dtree))
print(recall_score(y_test,result_dtree))
print(f1_score(y_test,result_dtree))
print(profit_score_function_unbalanced(y_test,result_dtree))
profit_matrix(y_test,result_dtree, 990, -10)

"""
################ GRID SEARCH NEAREST CENTROID
nc_grid = NearestCentroid()
params_cnn ={"metric":['euclidean', 'manhattan']}
best_model = grid_search_model(model=nc_grid, features=X_train, target=y_train, positive_label=1, parameters=params_cnn, fit_params=None, score="f1", folds=10)
#best_cnn_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
result_cnn = predict_general_model_results(best_model,x_test=X_test)
confusion_matrix_report(y_test,result_cnn)
print(accuracy_score(y_test,result_cnn))
print(precision_score(y_test,result_cnn))
print(recall_score(y_test,result_cnn))
print(f1_score(y_test,result_cnn))
"""
## test KNN
#params_knn = {'n_neighbors':3, 'weights' : "uniform", 'algorithm':"auto", 'leaf_size':30, 'p':2, 'metric':"minkowski", 'metric_params':None, 'n_jobs':None}
#knn_model = train_knn(params_knn, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

## test Decision_Trees
#params_dt = {'criterion':'gini', 'splitter':'best', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_decrease':0.0, 'min_impurity_split':None, 'class_weight':None, 'presort':False}
#dt_model = train_decision_tree(params_dt, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)


# ## GridSearch NaiveBayes
# params_nb_grid = {'priors':[None], 'var_smoothing':[1e-10, 1e-9, 1e-5, 1e-15, 1e-15]}
# nb = GaussianNB()
# best_model = grid_search_model(model=nb, features=X_train, target=y_train, positive_label=1, parameters=params_nb_grid, fit_params=None, score="precision", folds=5)
# best_nb_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_nb = predict_general_model_results(best_nb_model,x_test=X_test)
# confusion_matrix_report(y_test,result_nb)
# print(accuracy_score(y_test,result_nb))
# print(precision_score(y_test,result_nb))
# print(recall_score(y_test,result_nb))


## test NaiveBayes
# params_nb = {'priors': None, 'var_smoothing':1e-10}
# naive_bayes_model = train_naive_bayes(params_nb, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
# result_nb= predict_general_model_results(naive_bayes_model,x_test=X_test)
# confusion_matrix_report(y_test,result_nb)
# print(accuracy_score(y_test,result_nb))
# print(precision_score(y_test,result_nb))
# print(recall_score(y_test,result_nb))


#### GridSearch ComplementNaiveBayes
# params_cnb_grid = {'alpha':[1.0,2.0], 'fit_prior':[True,False], 'class_prior':[None], 'norm':[False,True]}
# cnb = ComplementNB()
# best_model = grid_search_model(model=cnb, features=X_train, target=y_train, positive_label=1, parameters=params_cnb_grid, fit_params=None, score="precision", folds=5)
# best_cnb_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_cnb = predict_general_model_results(best_cnb_model,x_test=X_test)
# confusion_matrix_report(y_test,result_cnb)
# print(accuracy_score(y_test,result_cnb))
# print(precision_score(y_test,result_cnb))
# print(recall_score(y_test,result_cnb))


## need to normalize before, not yet available
## test Complement NB
# params_nb = {'alpha':1.0, 'fit_prior':True, 'class_prior': None, 'norm':False}
# naive_bayes_model = train_complement_naiveBayes(params_nb, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)


## GridSearch BernoulliNaiveBayes
# params_bnb_grid = {'alpha':[1.0,2.0,1.5,3.0], 'binarize':[0.0], 'fit_prior':[False,True]}
# bnb = BernoulliNB()
# best_model = grid_search_model(model=bnb, features=X_train, target=y_train, positive_label=1, parameters=params_bnb_grid, fit_params=None, score="precision", folds=5)
# best_nb_model = train_general_model(best_model, x_train=X_train, y_train=y_train, n_folds=10, fit_params = None, random_state=123, stratified=True, i=0, shuffle=True)
# result_bnb = predict_general_model_results(best_nb_model,x_test=X_test)
# confusion_matrix_report(y_test,result_bnb)
# print(accuracy_score(y_test,result_bnb))
# print(precision_score(y_test,result_bnb))
# print(recall_score(y_test,result_bnb))

## test BernoulliNaiveBayes
# params_bnb = {'alpha':1.0, 'binarize':0.0, 'fit_prior':True, 'class_prior':None}
# b_naive_bayes_model = train_Bernoulli_NaiveBayes(params_bnb, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
# result_bnb= predict_general_model_results(b_naive_bayes_model,x_test=X_test)
# confusion_matrix_report(y_test,result_bnb)
# print(accuracy_score(y_test,result_bnb))
# print(precision_score(y_test,result_bnb))
# print(recall_score(y_test,result_bnb))

## Test Nearest Centroid
#params_nearest_centroid = {'metric':'manhattan'}
#nearest_cetroid_model = train_nearest_centroid(params_nearest_centroid, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)
# result_nc= predict_general_model_results(nearest_centroid_model,x_test=X_test)
# confusion_matrix_report(y_test,result_nc)
# print(accuracy_score(y_test,result_nc))
# print(precision_score(y_test,result_nc))
# print(recall_score(y_test,result_nc))

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