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


X_preprocessed_one_hot = data_preprocessing(data_set=X_full,
                                    columns_to_drop=['day_of_week', 'pdays', 'poutcome', 'duration'],
                                    columns_to_onehot=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                    columns_to_dummy=[],
                                    columns_to_label=[],
                                    normalise=True)

X_preprocessed_label = data_preprocessing(data_set=X_full,
                                    columns_to_drop=['day_of_week', 'pdays', 'poutcome', 'duration'],
                                    columns_to_onehot=[],
                                    columns_to_dummy=[],
                                    columns_to_label=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                    normalise=True)


X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(X_preprocessed_one_hot, y_full, test_size=0.20, random_state=42, stratify=y_full)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_preprocessed_label, y_full, test_size=0.20, random_state=42, stratify=y_full)


y_full.replace(('yes', 'no'), (1, 0), inplace=True)
y_test_oh.replace(('yes', 'no'), (1, 0), inplace=True)
y_test_l.replace(('yes', 'no'), (1, 0), inplace=True)
y_train_oh.replace(('yes', 'no'), (1, 0), inplace=True)
y_train_l.replace(('yes', 'no'), (1, 0), inplace=True)

X_train_balanced_oh, y_train_balanced_oh = data_balancing(X_train_oh, y_train_oh)
X_train_balanced_l, y_train_balanced_l = data_balancing(X_train_l, y_train_l)

print(X_train_oh.shape)
print(X_test_oh.shape)
print(y_train_oh.shape)
print(y_test_oh.shape)
print("BALANCED:")
print(X_train_balanced_oh.shape)
print(y_train_balanced_oh.shape)
print(y_full.value_counts())
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
    "gamma": [0.05, 0.5],
    "booster": ['gblinear'],
    "max_depth": [3, 5, 7, 9],
    "min_child_weight": [1, 3 ,5, 7],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "reg_lambda": [0.01, 0.1],
    "reg_alpha": [0, 0.1],
    "learning_rate": [0.1, 0.01],
    "n_estimators": [100],
    "objective": ["binary:logistic"],
    "nthread": [-1],
    "seed": [27]
    }

best_xgb = search_best_params_and_evaluate_general_model(classifier="XGBoost",
                                                         X_train = X_train_balanced_l,
                                                         y_train = y_train_balanced_l,
                                                         X_test = X_test_l,
                                                         y_test=y_test_l,
                                                         parameter_dict=params_xgb,
                                                         n_folds=5
                                                         )

y_pred_full = best_xgb.predict(X_preprocessed_label)
y_pred_test = best_xgb.predict(X_test_l)

print("Whole dataset score:")
print(profit_score_function(y_full, y_pred_full))
print("Confusion")
confusion_matrix_report(y_full, y_pred_full)
print("Acc")
print(accuracy_score(y_full, y_pred_full))
print("Precision")
print(precision_score(y_full, y_pred_full))
print("Recall")
print(recall_score(y_full, y_pred_full))
print("F1")
print(f1_score(y_full, y_pred_full))

print("Test dataset score:")
print(profit_score_function(y_test_l, y_pred_test))
print("Confusion")
confusion_matrix_report(y_test_l, y_pred_test)
print("Acc")
print(accuracy_score(y_test_l, y_pred_test))
print("Precision")
print(precision_score(y_test_l, y_pred_test))
print("Recall")
print(recall_score(y_test_l, y_pred_test))
print("F1")
print(f1_score(y_test_l, y_pred_test))