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

X_train_balanced, y_train_balanced = data_balancing(X_train, y_train)

y_train.replace(('yes', 'no'), (1, 0), inplace=True)

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

params_knn = {"n_neighbors":[16, 32],
#              "algorithm":['auto', 'ball_tree','kd_tree', 'brute'],
#              "metric":['euclidean', 'manhattan', 'minkowski', 'chebyshev']
              "p": [2]
              }

best_knn = search_best_params_and_evaluate_general_model(classifier="KNN",
                                                         X_train = X_train,
                                                         y_train = y_train,
                                                         X_test = X_test,
                                                         y_test=y_test,
                                                         parameter_dict=params_knn,
                                                         n_folds=10
                                                         )

######################################### GAUSSIAN NAIVE BAYES #########################################################

######################################### COMPLEMENT NB ################################################################

######################################### DECISION TREE ################################################################

######################################### XGBOOST ######################################################################

params_xgb = {
    "gamma": [1],
    "booster": ['gbtree'],
    "max_depth": [12],
    "min_child_weight": [3],
    "subsample": [0.6],
    "colsample_bytree": [1],
    "reg_lambda": [1],
    "reg_alpha": [1],
    "learning_rate": [0.01],
    "n_estimators": [100],
    "objective": ["binary:logistic"],
    "nthread": [-1],
    "seed": [27]
    }

best_xgb = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                         X_train = X_train,
                                                         y_train = y_train,
                                                         X_test = X_test,
                                                         y_test=y_test,
                                                         parameter_dict=params_xgb,
                                                         n_folds=10
                                                         )

