from ModelTraining import *
from DataPreProcessing import *
from sklearn.model_selection import train_test_split
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

######################################### DATA LOADING #################################################################

# Print all columns. Dont hide any.
pd.set_option('display.max_columns', None)

# NOTE: Adjust relative file path to your file system
bankingcalldata = pd.read_csv('bank-additional-full.csv', sep=';')

print('Full dataset shape: ')
print(bankingcalldata.shape)

######################################### PREPROCESSING ################################################################

# Check missing values
check_missing_values(bankingcalldata)

# Split X and y
X_full = bankingcalldata.drop('y', axis=1)
y_full = bankingcalldata['y']
y_full.replace(('yes', 'no'), (1, 0), inplace=True)

# Create new features
X_full = not_contacted(X_full)
X_full = contacted_last_9_months(X_full)
X_full = campaign_split(X_full)
X_full = elder_person(X_full)
X_full = is_student(X_full)
X_full = cellular_contact(X_full)
X_full = euribor_bin(X_full)
X_full = in_education(X_full)

# Apply binning
X_full['age'] = bin_age(X_full).astype('object')
X_full['duration'] = bin_duration(X_full).astype('object')

X_preprocessed_one_hot = data_preprocessing(data_set=X_full,
                                            columns_to_drop=['duration', 'day_of_week', 'poutcome', 'pdays', 'campaign'],
                                            columns_to_onehot=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                            columns_to_dummy=[],
                                            columns_to_label=[],
                                            normalise=True)

X_preprocessed_dummies = data_preprocessing(data_set=X_full,
                                            columns_to_drop=['duration', 'day_of_week', 'poutcome', 'pdays', 'campaign'],
                                            columns_to_onehot=[],
                                            columns_to_dummy=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                            columns_to_label=[],
                                            normalise=True)

X_preprocessed_label = data_preprocessing(data_set=X_full,
                                          columns_to_drop=['duration', 'day_of_week', 'poutcome', 'pdays', 'campaign'],
                                          columns_to_onehot=[],
                                          columns_to_dummy=[],
                                          columns_to_label=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                          normalise=True)


X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_preprocessed_one_hot, y_full, test_size=0.20, random_state=42, stratify=y_full)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_preprocessed_label, y_full, test_size=0.20, random_state=42, stratify=y_full)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_preprocessed_dummies, y_full, test_size=0.20, random_state=42, stratify=y_full)

X_train_balanced_o, y_train_balanced_o = data_balancing(X_train_o, y_train_o)
X_train_balanced_l, y_train_balanced_l = data_balancing(X_train_l, y_train_l)
X_train_balanced_d, y_train_balanced_d = data_balancing(X_train_d, y_train_d)

print("ONE-HOT ENCODED: \n")
print(X_train_o.shape)
print(X_test_o.shape)
print(y_train_o.shape)
print(y_test_o.shape)
print()
print("BALANCED ONE-HOT: \n")
print(X_train_balanced_o.shape)
print(y_train_balanced_o.shape)
print()
print("DUMMY ENCODED: \n")
print(X_train_d.shape)
print(X_test_d.shape)
print(y_train_d.shape)
print(y_test_d.shape)
print()
print("BALANCED DUMMY ENCODED: \n")
print(X_train_balanced_d.shape)
print(y_train_balanced_d.shape)
print()
print("LABEL ENCODED: \n")
print(X_train_d.shape)
print(X_test_d.shape)
print(y_train_d.shape)
print(y_test_d.shape)
print()
print("BALANCED LABEL ENCODED: \n")
print(X_train_balanced_d.shape)
print(y_train_balanced_d.shape)
print()
print("TARGET LABEL DISTRIBUTION: \n")
print(y_full.value_counts())
print()

######################################### GRID SEARCH ##################################################################

######################################### LOGISTIC REGRESSION ##########################################################

params_log = {"penalty": ["l2"],
              "C": [0.001, 0.01, 0.1, 1, 10, 100],
              'class_weight': [None, 'balanced'],
              'solver': ['sag', 'newton-cg', 'lbfgs']
              }
best_log_unbalanced = search_best_params_and_evaluate_general_model(classifier="LogisticRegression",
                                                                    X_train=X_train_d,
                                                                    y_train=y_train_d,
                                                                    X_test=X_test_d,
                                                                    y_test=y_test_d,
                                                                    parameter_dict=params_log,
                                                                    n_folds=5
                                                                    )

best_log_balanced = search_best_params_and_evaluate_general_model(classifier="LogisticRegression",
                                                                  X_train=X_train_balanced_d,
                                                                  y_train=y_train_balanced_d,
                                                                  X_test=X_test_d,
                                                                  y_test=y_test_d,
                                                                  parameter_dict=params_log,
                                                                  n_folds=5
                                                                  )

######################################### RANDOM FOREST ################################################################

params_rf = {'n_estimators': [10, 50, 100, 200, 500, 800, 1000],
             'max_depth': [2, 5, 10, 20],
             'min_samples_split': [2, 3, 4],
             'min_samples_leaf': [2, 3, 4],
             'max_features': [1, 2, 3, 4, 5, 6, 7, None],
             'random_state': [123]
             }

best_rf_unbalanced = search_best_params_and_evaluate_general_model(classifier="RandomForest",
                                                                   X_train=X_train_d,
                                                                   y_train=y_train_d,
                                                                   X_test=X_test_d,
                                                                   y_test=y_test_d,
                                                                   parameter_dict=params_rf,
                                                                   n_folds=5
                                                                   )

best_rf_balanced = search_best_params_and_evaluate_general_model(classifier="RandomForest",
                                                                 X_train=X_train_balanced_d,
                                                                 y_train=y_train_balanced_d,
                                                                 X_test=X_test_d,
                                                                 y_test=y_test_d,
                                                                 parameter_dict=params_rf,
                                                                 n_folds=5
                                                                 )

######################################### BERNOULLI NAIVE BAYES ########################################################

params_bnb = {'alpha': [1.0, 2.0, 1.5, 3.0],
              'binarize': [0.0],
              'fit_prior': [False, True]
              }

best_bnb_unbalanced = search_best_params_and_evaluate_general_model(classifier="BernoulliNB",
                                                                    X_train=X_train_d,
                                                                    y_train=y_train_d,
                                                                    X_test=X_test_d,
                                                                    y_test=y_test_d,
                                                                    parameter_dict=params_bnb,
                                                                    n_folds=5
                                                                    )

best_bnb_balanced = search_best_params_and_evaluate_general_model(classifier="BernoulliNB",
                                                                  X_train=X_train_balanced_d,
                                                                  y_train=y_train_balanced_d,
                                                                  X_test=X_test_d,
                                                                  y_test=y_test_d,
                                                                  parameter_dict=params_bnb,
                                                                  n_folds=5
                                                                  )

######################################### NEAREST CENTROID #############################################################

params_nc = {'metric': ['euclidean', 'manhattan']
             }

best_nc_unbalanced = search_best_params_and_evaluate_general_model(classifier="NearestCentroid",
                                                                   X_train=X_train_d,
                                                                   y_train=y_train_d,
                                                                   X_test=X_test_d,
                                                                   y_test=y_test_d,
                                                                   parameter_dict=params_nc,
                                                                   n_folds=5
                                                                   )

best_nc_balanced = search_best_params_and_evaluate_general_model(classifier="NearestCentroid",
                                                                 X_train=X_train_balanced_d,
                                                                 y_train=y_train_balanced_d,
                                                                 X_test=X_test_d,
                                                                 y_test=y_test_d,
                                                                 parameter_dict=params_nc,
                                                                 n_folds=5
                                                                 )

######################################### K-NEAREST NEIGHBORS ##########################################################

params_knn = {'n_neighbors': [2, 4, 8, 16, 32, 64],
              'algorithm': ['auto', 'ball_tree'],
              'metric': ['euclidean', 'manhattan'],
              'p': [2]
              }

best_knn_unbalanced = search_best_params_and_evaluate_general_model(classifier="KNN",
                                                                    X_train=X_train_d,
                                                                    y_train=y_train_d,
                                                                    X_test=X_test_d,
                                                                    y_test=y_test_d,
                                                                    parameter_dict=params_knn,
                                                                    n_folds=5
                                                                    )


best_knn_balanced = search_best_params_and_evaluate_general_model(classifier="KNN",
                                                                  X_train=X_train_balanced_d,
                                                                  y_train=y_train_balanced_d,
                                                                  X_test=X_test_d,
                                                                  y_test=y_test_d,
                                                                  parameter_dict=params_knn,
                                                                  n_folds=5
                                                                  )

######################################### GAUSSIAN NAIVE BAYES #########################################################

params_gnb = {}

best_gnb_unbalanced = search_best_params_and_evaluate_general_model(classifier="GaussianNB",
                                                                    X_train=X_train_d,
                                                                    y_train=y_train_d,
                                                                    X_test=X_test_d,
                                                                    y_test=y_test_d,
                                                                    parameter_dict=params_gnb,
                                                                    n_folds=5
                                                                    )

best_gnb_balanced = search_best_params_and_evaluate_general_model(classifier="GaussianNB",
                                                                  X_train=X_train_balanced_d,
                                                                  y_train=y_train_balanced_d,
                                                                  X_test=X_test_d,
                                                                  y_test=y_test_d,
                                                                  parameter_dict=params_gnb,
                                                                  n_folds=5
                                                                  )

######################################### COMPLEMENT NB ################################################################

params_cnb = {'alpha': [1.0,2.0],
              'fit_prior': [True,False],
              'class_prior': [None],
              'norm': [False,True]
              }

best_cnb_unbalanced = search_best_params_and_evaluate_general_model(classifier="ComplementNB",
                                                                    X_train=X_train_d,
                                                                    y_train=y_train_d,
                                                                    X_test=X_test_d,
                                                                    y_test=y_test_d,
                                                                    parameter_dict=params_gnb,
                                                                    n_folds=5
                                                                    )

best_cnb_balanced = search_best_params_and_evaluate_general_model(classifier="ComplementNB",
                                                                  X_train=X_train_balanced_d,
                                                                  y_train=y_train_balanced_d,
                                                                  X_test=X_test_d,
                                                                  y_test=y_test_d,
                                                                  parameter_dict=params_gnb,
                                                                  n_folds=5
                                                                  )

######################################### DECISION TREE ################################################################

params_dt = {'criterion': ['gini', 'entropy'],
             'splitter': ['best'],
             'max_depth': [2, 5, 10, None],
             'min_samples_split': [2, 4, 6, 10],
             'min_samples_leaf': [15, 20, 25],
             'min_weight_fraction_leaf': [0.0, 0.15, 0.2],
             'random_state': [123],
             'min_impurity_decrease': [0.0]
             }

best_dt_unbalanced = search_best_params_and_evaluate_general_model(classifier="DecisionTree",
                                                                   X_train=X_train_d,
                                                                   y_train=y_train_d,
                                                                   X_test=X_test_d,
                                                                   y_test=y_test_d,
                                                                   parameter_dict=params_dt,
                                                                   n_folds=5
                                                                   )

best_dt_balanced = search_best_params_and_evaluate_general_model(classifier="DecisionTree",
                                                                 X_train=X_train_balanced_d,
                                                                 y_train=X_train_balanced_d,
                                                                 X_test=X_test_d,
                                                                 y_test=X_test_d,
                                                                 parameter_dict=params_dt,
                                                                 n_folds=5
                                                                 )

######################################### SVM ##########################################################################

params_svm = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'class_weight': ["balanced", None]
              }

best_svm_unbalanced = search_best_params_and_evaluate_general_model(classifier="SupportVectorMachine",
                                                                    X_train=X_train_d,
                                                                    y_train=y_train_d,
                                                                    X_test=X_test_d,
                                                                    y_test=y_test_d,
                                                                    parameter_dict=params_svm,
                                                                    n_folds=5
                                                                    )

best_svm_balanced = search_best_params_and_evaluate_general_model(classifier="SupportVectorMachine",
                                                                  X_train=X_train_balanced_d,
                                                                  y_train=X_train_balanced_d,
                                                                  X_test=X_test_d,
                                                                  y_test=X_test_d,
                                                                  parameter_dict=params_svm,
                                                                  n_folds=5
                                                                  )


######################################### XGBOOST ######################################################################

params_xgb = {'gamma': [0.05, 0.5],
              'booster': ['gblinear'],
              'max_depth': [3, 5, 7, 9],
              'min_child_weight': [1, 3 ,5, 7],
              'subsample': [0.6, 0.7, 0.8],
              'colsample_bytree': [0.6, 0.7, 0.8],
              'reg_lambda': [0.01, 0.1],
              'reg_alpha': [0, 0.1],
              'learning_rate': [0.1, 0.01],
              'n_estimators': [100],
              'objective': ["binary:logistic"],
              'nthread': [-1],
              'seed': [27]
              }

best_xgb_unbalanced = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                                    X_train=X_train_l,
                                                                    y_train=y_train_l,
                                                                    X_test=X_test_l,
                                                                    y_test=y_test_l,
                                                                    parameter_dict=params_xgb,
                                                                    n_folds=5
                                                                    )

best_xgb_balanced = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                                  X_train=X_train_balanced_l,
                                                                  y_train=y_train_balanced_l,
                                                                  X_test=X_test_l,
                                                                  y_test=y_test_l,
                                                                  parameter_dict=params_xgb,
                                                                  n_folds=5
                                                                  )

######################################### ENSEMBLE XGBOOST #############################################################

train_probas = pd.read_csv("1st_level_probs_train.csv")
test_probas = pd.read_csv("1st_level_probs_test.csv")
y_train_e = pd.read_csv("1st_level_y_train.csv")
y_test_e = pd.read_csv("1st_level_y_test.csv")

full_probas = train_probas.append(test_probas)
y_probas = y_train_e.append(y_test_e)

train_probas = train_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)
test_probas = test_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)
full_probas = full_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)

x_train_probas_balanced, y_train_probas_balanced = data_balancing(train_probas, y_train_e)

params_xgb_ensemble = {
    "gamma": [0.05, 1],
    "booster": ['gbtree'],
    "max_depth": [3, 25],
    "min_child_weight": [1, 7],
    "subsample": [0.6, 1],
    "colsample_bytree": [0.6, 1],
    "reg_lambda": [0.01, 1],
    "reg_alpha": [0, 1],
    "learning_rate": [0.1, 0.01],
    "n_estimators": [100],
    "objective": ["binary:logistic"],
    "nthread": [-1],
    "seed": [27]
    }

best_xgb_e_unbalanced = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                                      X_full=full_probas,
                                                                      y_full=y_full,
                                                                      X_train=train_probas,
                                                                      y_train=y_train_e,
                                                                      X_test=test_probas,
                                                                      y_test=y_test_e,
                                                                      parameter_dict=params_xgb_ensemble,
                                                                      n_folds=5
                                                                      )

best_xgb_e_balanced = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                                    X_full=full_probas,
                                                                    y_full=y_full,
                                                                    X_train=x_train_probas_balanced,
                                                                    y_train=y_train_probas_balanced,
                                                                    X_test=test_probas,
                                                                    y_test=y_test_e,
                                                                    parameter_dict=params_xgb_ensemble,
                                                                    n_folds=5
                                                                    )

######################################### ENSEMBLE RANDOM FOREST #######################################################

params_rf_ensemble = {'n_estimators': [10, 50, 100, 200, 500, 800, 1000],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 3, 4],
                      'min_samples_leaf': [2, 3, 4],
                      'max_features': [1, 2, 3, 4, 5],
                      'random_state': [123]
                      }

best_rf_e_unbalanced = search_best_params_and_evaluate_general_model(classifier="Random Forest",
                                                                     X_full=full_probas,
                                                                     y_full=y_full,
                                                                     X_train=x_train_probas_balanced,
                                                                     y_train=y_train_probas_balanced,
                                                                     X_test=test_probas,
                                                                     y_test=y_test_e,
                                                                     parameter_dict=params_rf_ensemble,
                                                                     n_folds=5
                                                                     )

best_rf_e_balanced = search_best_params_and_evaluate_general_model(classifier="Random Forest",
                                                                   X_full=full_probas,
                                                                   y_full=y_full,
                                                                   X_train=train_probas,
                                                                   y_train=y_train_e,
                                                                   X_test=test_probas,
                                                                   y_test=y_test_e,
                                                                   parameter_dict=params_rf_ensemble,
                                                                   n_folds=5
                                                                   )