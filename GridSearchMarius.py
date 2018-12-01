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
#X_full['age'] = bin_age(X_full).astype('object')
#X_full['duration'] = bin_duration(X_full).astype('object')
#X_full['pmonths'] = bin_pdays(X_full).astype('object')

# Create new features
# X_full = not_contacted(X_full)

X_preprocessed = data_preprocessing(data_set=X_full,
                                    columns_to_drop=['duration'],
                                    columns_to_onehot=[],
                                    columns_to_dummy=[],
                                    columns_to_label=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week','previous','poutcome'],
                                    normalise=True)

print(y_full.head())

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_full, test_size=0.20, random_state=42, stratify=y_full)

y_train.replace(('yes', 'no'), (1, 0), inplace=True)
y_test.replace(('yes', 'no'), (1, 0), inplace=True)
y_full.replace(('yes', 'no'), (1, 0), inplace=True)

X_train_balanced, y_train_balanced = data_balancing(X_train, y_train)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("BALANCED:")
print(X_train_balanced.shape)
print(y_train_balanced.shape)
print(y_full.value_counts())

######################################### GRID SEARCH ##################################################################

######################################### LOGISTIC REGRESSION ##########################################################

######################################### RANDOM FOREST ################################################################

######################################### BERNOULLI NAIVE BAYES ########################################################

######################################### MLP ##########################################################################

######################################### NEAREST CENTROID #############################################################

######################################### K-NEAREST NEIGHBORS ##########################################################

######################################### GAUSSIAN NAIVE BAYES #########################################################

######################################### COMPLEMENT NB ################################################################

######################################### DECISION TREE ################################################################

######################################### XGBOOST ######################################################################
'''
params_xgb = {
    "gamma": [0.5, 1],
    "booster": ['gbtree'],
    "max_depth": [9, 12, 25],
    "min_child_weight": [3,5, 7],
    "subsample": [0.8, 0.9, 1],
    "colsample_bytree": [0.8, 0.9,1],
    "reg_lambda": [0.01,0.1],
    "reg_alpha": [0, 0.5],
    "learning_rate": [0.1, 0.05],
    "n_estimators": [100],
    "objective": ["binary:logistic"],
    "nthread": [-1],
    "seed": [27]
    }

best_xgb = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                         X_full=X_preprocessed,
                                                         y_full=y_full,
                                                         X_train=X_train,
                                                         y_train=y_train,
                                                         X_test=X_test,
                                                         y_test=y_test,
                                                         parameter_dict=params_xgb,
                                                         n_folds=5
                                                         )
'''
######################################### ENSEMBLE XGBOOST #############################################################

train_probas = pd.read_csv("1st_level_probs_train.csv")
test_probas = pd.read_csv("1st_level_probs_test.csv")

train_probas = train_probas.set_index(y_train.index)

train_probas = train_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)
test_probas = test_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)

x_train_probas_balanced, y_train_probas_balanced = data_balancing(train_probas, y_train)


params_xgb = {
    "gamma": [0.05, 1],
    "booster": ['gbtree', 'gblinear', 'dart'],
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

best_xgb = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                         X_full=X_preprocessed,
                                                         y_full=y_full,
                                                         X_train=x_train_probas_balanced,
                                                         y_train=y_train_probas_balanced,
                                                         X_test=test_probas,
                                                         y_test=y_test,
                                                         parameter_dict=params_xgb,
                                                         n_folds=5
                                                         )

######################################### ENSEMBLE RANDOM FOREST #######################################################


