# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

from ModelTraining import *
from dataPreProcessing_Soumya import *

### Data Loading #####

#Print all rows and columns. Dont hide any.
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

bankingcalldata = pd.read_csv('C:/Users/hroed/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')

print('Full dataset shape: ')
print(bankingcalldata.shape)

from sklearn.model_selection import train_test_split


######################################### PREPROCESSING ################################################################

# Check missing values
check_missing_values(bankingcalldata)

# Split X and y
X_full = bankingcalldata.drop('y', axis=1)
y_full = bankingcalldata['y']
y_full.replace(('yes', 'no'), (1, 0), inplace=True)

# Apply binning
X_full['age'] = bin_age(X_full).astype('object')
#X_full['duration'] = bin_duration(X_full).astype('object')
X_full['pmonths'] = bin_pdays(X_full).astype('object')

# Create new features
X_full = not_contacted(X_full)


X_preprocessed = data_preprocessing(data_set=X_full,
                                    columns_to_drop=["duration", "day_of_week",'poutcome', "pdays"],
                                    columns_to_onehot=['month'],
                                    columns_to_dummy=["age", "marital", "education", "default", 'housing', 'loan', 'contact', "pmonths"],
                                    columns_to_label=["job"],
                                    normalise=True)

print(y_full.head())

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_full, test_size=0.20, random_state=42, stratify=y_full)

X_train, y_train = data_balancing(X_train, y_train)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("BALANCED:")
#print(X_train_balanced.shape)
#print(y_train_balanced.shape)
print(y_full.value_counts())

######################################### GRID SEARCH ##################################################################

######################################### LOGISTIC REGRESSION ##########################################################
# params_log = {"penalty":["l2"],
#               "C":[0.001, 0.01, 0.1, 1, 10, 100],
#               'class_weight': [None,'balanced'],
#               'solver':['sag','newton-cg','lbfgs']
#               }
# best_log = search_best_params_and_evaluate_general_model(classifier="LogisticRegression",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_log,
#                                                          n_folds=5
#                                                          )
######################################### RANDOM FOREST ################################################################
# params_rdf = {'n_estimators':[500],
#               'max_depth':[24, 25, 26],
#               'min_samples_split':[2, 3],
#               'min_samples_leaf': [4, 5, 6],
#               'max_features': [None],
#               'random_state': [42]
#               }
#
# best_rdf = search_best_params_and_evaluate_general_model(classifier="RandomForest",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_rdf,
#                                                          n_folds=5
#                                                          )
######################################### BERNOULLI NAIVE BAYES ########################################################
# params_bnb = {'alpha':[1.0,2.0,1.5,3.0],
#               'binarize':[0.0],
#               'fit_prior':[False,True]}
#
# best_bnb = search_best_params_and_evaluate_general_model(classifier="BernoulliNB",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_bnb,
#                                                          n_folds=5
#                                                          )
######################################### MLP ##########################################################################

######################################### NEAREST CENTROID #############################################################
# params_ncc ={"metric":['euclidean', 'manhattan']
#              }
#
# best_ncc = search_best_params_and_evaluate_general_model(classifier="NearestCentroid",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_ncc,
#                                                          n_folds=5
#                                                          )
######################################### K-NEAREST NEIGHBORS ##########################################################
# params_knn = {"n_neighbors":[2, 4, 8, 16, 32, 64],
#               "algorithm":['auto', 'ball_tree'],
#               "metric":['euclidean', 'manhattan'],
#               "p": [2, 3]
#               }
# best_knn = search_best_params_and_evaluate_general_model(classifier="KNN",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_knn,
#                                                          n_folds=5
#                                                          )
######################################### GAUSSIAN NAIVE BAYES #########################################################
# params_gnb = {}
# best_gnb = search_best_params_and_evaluate_general_model(classifier="GaussianNB",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_gnb,
#                                                          n_folds=5
#                                                          )
######################################### COMPLEMENT NB ################################################################

######################################### DECISION TREE ################################################################
params_dtree = {'criterion':['gini', 'entropy'],
                'splitter':['best'],
                'max_depth':[2, 5, 10, None],
                'min_samples_split':[2, 4, 6, 10],
                'min_samples_leaf':[15, 20, 25],
                'min_weight_fraction_leaf':[0.0, 0.15, 0.2],
                'random_state':[123],
                'min_impurity_decrease':[0.0, 0.1, 0.2]
                }
best_dtree = search_best_params_and_evaluate_general_model(classifier="DecisionTree",
                                                         X_train = X_train,
                                                         y_train = y_train,
                                                         X_test = X_test,
                                                         y_test=y_test,
                                                         parameter_dict=params_dtree,
                                                         n_folds=5
                                                         )
######################################### XGBOOST ######################################################################

# params_xgb = {
#     "gamma": [0.05, 1],
#     "booster": ['gbtree', 'gblinear', 'dart'],
#     "max_depth": [3, 25],
#     "min_child_weight": [1, 7],
#     "subsample": [0.6, 1],
#     "colsample_bytree": [0.6, 1],
#     "reg_lambda": [0.01, 1],
#     "reg_alpha": [0, 1],
#     "learning_rate": [0.1, 0.01],
#     "n_estimators": [100],
#     "objective": ["binary:logistic"],
#     "nthread": [-1],
#     "seed": [27]
#     }
#
# best_xgb = search_best_params_and_evaluate_general_model(classifier="XGBoost",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_xgb,
#                                                          n_folds=5
#                                                          )
######################################### SUPPORT VECTOR MACHINES ######################################################
# params_svm = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
#               'class_weight':["balanced", None]
#               }
#
# best_svm = search_best_params_and_evaluate_general_model(classifier="SupportVectorMachine",
#                                                          X_train = X_train,
#                                                          y_train = y_train,
#                                                          X_test = X_test,
#                                                          y_test=y_test,
#                                                          parameter_dict=params_svm,
#                                                          n_folds=5
#                                                          )
