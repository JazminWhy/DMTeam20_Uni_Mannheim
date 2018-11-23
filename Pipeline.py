# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

from ModelTraining import *
from dataPreProcessing_Soumya import *

### Data Loading #####

#Print all rows and columns. Dont hide any.
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

bankingcalldata = pd.read_csv('/Users/mariusbock/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')

print('Full dataset shape: ')
print(bankingcalldata.shape)

from sklearn.model_selection import train_test_split


######################################### PREPROCESSING ################################################################

# Check missing values
check_missing_values(bankingcalldata)

# Split X and y
X_full = bankingcalldata.drop('y', axis=1)
y_full = bankingcalldata['y']

# Apply binning
X_full['age'] = bin_age(X_full).astype('object')
X_full['duration'] = bin_duration(X_full).astype('object')
X_full['pmonths'] = bin_pdays(X_full).astype('object')

# Create new features
X_full = not_contacted(X_full)


X_preprocessed = data_preprocessing(data_set=X_full,
                                    columns_to_drop=['day_of_week', 'pdays', 'poutcome'],
                                    columns_to_onehot=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month','duration'],
                                    columns_to_dummy=[],
                                    columns_to_label=[],
                                    normalise=True)

print(X_preprocessed)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_full, test_size=0.20, random_state=42, stratify=y_full)

y_train.replace(('yes', 'no'), (1, 0), inplace=True)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_full.value_counts())

######################################### MODEL TRAINING ###############################################################

######################################### 1ST LEVEL TRAINING ###########################################################

### Base Classifiers ###

classifiers = {
    'naive_bayes': False,
    'nearest_centroid': False,
    'knn': False,
    'decision_tree': False,
    'rule_learner': False,
    'xgboost': True,
    'lightgbm': False,
    'neural_net': False,
}

if classifiers['naive_bayes']:
    print('Training Naive Bayes')
    params_nb = {
        'priors': None,
        'var_smoothing':1e-10
    }

    naive_bayes_model = train_naive_bayes(params_nb, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

if classifiers['nearest_centroid']:
    print('Training Nearest Centroid')

    params_nearest_centroid = {
        'metric':'manhattan'
        }

    nearest_centroid_model = train_nearest_centroid(params_nearest_centroid, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

if classifiers['knn']:
    print('Training KNN')

    params_knn = {
        'n_neighbors':3,
        'weights' : "uniform",
        'algorithm':"auto",
        'leaf_size':30,
        'p':2,
        'metric':"minkowski",
        'metric_params':None,
        'n_jobs':None}

    knn_model = train_knn(params_knn, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

if classifiers['decision_tree']:
    print('Training Decision Tree')

    params_dt = {
        'criterion':'gini',
        'splitter':'best',
        'max_depth':None,
        'min_samples_split':2,
        'min_samples_leaf':1,
        'min_weight_fraction_leaf':0.0,
        'max_features':None,
        'random_state':None,
        'max_leaf_nodes':None,
        'min_impurity_decrease':0.0,
        'min_impurity_split':None,
        'class_weight':None,
        'presort':False
    }

    decision_tree_model = train_decision_tree(params_dt, fit_params=None, x_train=X_train, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

if classifiers['rule_learner']:
    print('Training Rule Learner')

    X_train_rules = X_train.drop('nr.employed', axis=1)
    X_train_rules = X_train_rules.drop('emp.var.rate', axis=1)
    X_train_rules = X_train_rules.drop('cons.conf.idx', axis=1)
    X_train_rules = X_train_rules.drop('cons.price.idx', axis=1)

    params_rule = {
        'max_depth_duplication':None,
        'n_estimators':10,
        'precision_min':0.2,
        'recall_min':0.01,
        'feature_names': list(X_train_rules.columns.values)
    }
    rule_learner_model = skope_rules(params_rule, fit_params=None, x_train=X_train_rules, y_train = y_train, n_folds=5, random_state=123, stratified=True, i=0, shuffle=True)

### Advanced Classifiers ###

if classifiers['xgboost']:
    print('Training XGBoost')

    xgb_params_1 = {
        'gamma': 1,
        'booster': 'gbtree',
        'max_depth': 12,
        'min_child_weight': 1,
        'subsample': 0.6,
        'colsample_bytree': 1,
        'reg_lambda': 1,
        'reg_alpha': 1,
        'learning_rate': 0.01,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'nthread': -1,
        'seed': 27
    }

    xgb_model_1 = train_xgb_model(params=xgb_params_1,
                                  x_train=X_train,
                                  y_train=y_train,
                                  n_folds=2,
                                  random_state=123
                                  )

if classifiers['lightgbm']:
    print('Training LightGBM')

    lgbm_params_1 = {
        'boosting_type': 'gbdt',
        'num_leaves': 1000,
        'max_depth': 10,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        # 'subsample_for_bin': 1,
        'objective': 'binary',
        # 'class_weight': 0,
        # 'min_split_gain': 0.8,
        # 'min_child_weight': 0.8,
        # 'min_child_samples':0.8,
        # 'subsample': 'binary:logistic',
        # 'subsample_freq': -1,
        # 'colsample_bytree': 1,
        # 'reg_alpha':1,
        # 'reg_lambda': 27,
        # 'random_state': 'logloss',
        # 'n_jobs': 'gain',
        # 'silent':np.nan,
        # 'importance_type':10,
    }

    lgbm_model_1 = train_lgbm_model(X_train, y_train,
                                                  params= lgbm_params_1,
                                                  n_folds=10,
                                                  early_stopping=50,
                                                  random_state=123)

if classifiers['neural_net']:
    print('Training Neural Net')

    nn_model_1 = train_nn_model(X_train, y_train,
                                              epochs=20,
                                              n_folds=10,
                                              random_state=123)

######################################### 2ND LEVEL TRAINING ###########################################################

## Create dataframe with models from 1st Level Training
first_level_models = []

if classifiers['naive_bayes']:
    first_level_models.append(naive_bayes_model)
if classifiers['nearest_centroid']:
    first_level_models.append(nearest_centroid_model)
if classifiers['knn']:
    first_level_models.append(knn_model)
if classifiers['decision_tree']:
    first_level_models.append(decision_tree_model)
if classifiers['rule_learner']:
    first_level_models.append(rule_learner_model)
if classifiers['xgboost']:
    first_level_models.append(xgb_model_1)
if classifiers['lightgbm']:
    first_level_models.append(lgbm_model_1)
if classifiers['neural_net']:
    first_level_models.append(nn_model_1)

print(first_level_models)

print('Training Ensemble using XGBoost')

ensemble_xgb_params_1 = {
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'max_depth': 5,
    # 'silent': False,
    # 'booster': 'gbtree',
    'min_child_weight': 1,
    # 'max_delta_step':10,
    # 'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # 'colsample_bylevel':0.8,
    'objective': 'binary:logistic',
    # 'nthread': -1,
    # 'scale_pos_weight': 1,
    # 'base_score':1,
    # 'random_state': 27,
    # importance_type: 'gain',
    # 'missing':np.nan,
    # 'reg_alpha':10,
    # 'reg_lambda': 10,
    # 'num_class': 1,
}

ensemble_model_xgb_1 = train_xgb_ensemble(models=first_level_models, y_train = y_train, x_train=X_train,
                                                        params=ensemble_xgb_params_1,
                                                        n_folds=10,
                                                        random_state=123
                                                        )


print('Training Ensemble using Neural Net')

######################################### MODEL EVALUATION #############################################################

