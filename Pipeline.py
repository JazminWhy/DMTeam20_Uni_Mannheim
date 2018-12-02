from ModelTraining import *
from DataPreProcessing import *
from sklearn.model_selection import train_test_split
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')


######################################### DISCLAIMER ###################################################################

# This class is the final pipeline. It trains various classifiers and finally trains 2 meta classifiers on the predicted
# probabilities of the first level classifiers. It incorporates all methods of the other classes and is the final
# product of our project.

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

# create preprocessed datasets (one-hot, dummy and label encoded)
X_preprocessed_one_hot = data_preprocessing(data_set=X_full,
                                            columns_to_drop=['duration', 'day_of_week','poutcome', 'pdays', 'campaign'],
                                            columns_to_onehot=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                            columns_to_dummy=[],
                                            columns_to_label=[],
                                            normalise=True)

X_preprocessed_dummies = data_preprocessing(data_set=X_full,
                                            columns_to_drop=['duration', 'day_of_week','poutcome', 'pdays', 'campaign'],
                                            columns_to_onehot=[],
                                            columns_to_dummy=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                            columns_to_label=[],
                                            normalise=True)

X_preprocessed_label = data_preprocessing(data_set=X_full,
                                          columns_to_drop=['duration', 'day_of_week','poutcome', 'pdays', 'campaign'],
                                          columns_to_onehot=[],
                                          columns_to_dummy=[],
                                          columns_to_label=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'],
                                          normalise=True)

# perform 80/20 train-test-split for each
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_preprocessed_one_hot, y_full,
                                                            test_size=0.20, random_state=42, stratify=y_full)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_preprocessed_dummies, y_full,
                                                            test_size=0.20, random_state=42, stratify=y_full)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_preprocessed_label, y_full,
                                                            test_size=0.20, random_state=42, stratify=y_full)

# create balanced train data for each
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

######################################### 1ST LEVEL TRAINING ###########################################################

# Dictionary to decide which ones to run; set value to False if you want the algorithm to be skipped
classifiers = {
    'g_naive_bayes': True,
    'b_naive_bayes': True,
    'c_naive_bayes': True,
    'nearest_centroid': True,
    'knn': True,
    'decision_tree': True,
    'logistic': True,
    'random_forest': True,
    'svm': True,
    'xgboost': True
}

# Base Classifiers
if classifiers['g_naive_bayes']:
    print('\n Training Gaussian Naive Bayes \n')

    params_gnb = {
        'priors': None,
        'var_smoothing': 1e-10
    }

    g_naive_bayes_model = train_naive_bayes(params_gnb,
                                            x_train=X_train_d,
                                            y_train=y_train_d,
                                            n_folds=10,
                                            random_state=123,
                                            stratified=True,
                                            shuffle=True
                                            )

    gnb_x_train_probas = g_naive_bayes_model.predict_proba(X_train_d)
    gnb_x_test_probas = g_naive_bayes_model.predict_proba(X_test_d)

    y_pred_full = g_naive_bayes_model.predict(X_preprocessed_dummies)
    y_pred_test = g_naive_bayes_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))


if classifiers['nearest_centroid']:
    print('\n Training Nearest Centroid \n')

    params_nearest_centroid = {
        'metric': 'euclidean'
        }

    nearest_centroid_model = train_nearest_centroid(params_nearest_centroid,
                                                    x_train=X_train_o,
                                                    y_train=y_train_o,
                                                    n_folds=10,
                                                    random_state=123,
                                                    stratified=True,
                                                    shuffle=True
                                                    )

    nc_x_train_preds = nearest_centroid_model.predict(X_train_o)
    nc_x_test_preds = nearest_centroid_model.predict(X_test_o)

    y_pred_full = nearest_centroid_model.predict(X_preprocessed_one_hot)
    y_pred_test = nearest_centroid_model.predict(X_test_o)

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
    print(profit_score_function(y_test_o, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_o, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_o, y_pred_test))
    print("Precision")
    print(precision_score(y_test_o, y_pred_test))
    print("Recall")
    print(recall_score(y_test_o, y_pred_test))
    print("F1")
    print(f1_score(y_test_o, y_pred_test))


if classifiers['knn']:
    print('\n Training KNN \n')

    params_knn = {'algorithm': 'ball_tree',
                  'metric': 'manhattan',
                  'n_neighbors': 64,
                  'p': 2
                  }

    knn_model = train_knn(params_knn,
                          x_train=X_train_balanced_d,
                          y_train=y_train_balanced_d,
                          n_folds=10,
                          random_state=123,
                          stratified=True,
                          shuffle=True
                          )

    knn_x_train_probas = knn_model.predict_proba(X_train_d)
    knn_x_test_probas = knn_model.predict_proba(X_test_d)

    y_pred_full = knn_model.predict(X_preprocessed_dummies)
    y_pred_test = knn_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))


if classifiers['decision_tree']:
    print('\n Training Decision Tree \n')

    params_dt = {'criterion': 'gini',
                 'max_depth': 2,
                 'min_impurity_decrease': 0.0,
                 'min_samples_leaf': 15,
                 'min_samples_split': 2,
                 'min_weight_fraction_leaf': 0.15,
                 'random_state': 123,
                 'splitter': 'best'
                 }

    decision_tree_model = train_decision_tree(params_dt,
                                              x_train=X_train_balanced_d,
                                              y_train=y_train_balanced_d,
                                              n_folds=10,
                                              random_state=123,
                                              stratified=True,
                                              shuffle=True
                                              )

    dt_x_train_probas = decision_tree_model.predict_proba(X_train_d)
    dt_x_test_probas = decision_tree_model.predict_proba(X_test_d)

    y_pred_full = decision_tree_model.predict(X_preprocessed_dummies)
    y_pred_test = decision_tree_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))


if classifiers['logistic']:
    print('\n Training Logistic Regression \n')

    params_log = {'C': 0.001,
                  'class_weight': None,
                  'penalty': 'l2',
                  'solver': 'sag'
                  }

    logistic_model = train_logistic(params_log,
                                    x_train=X_train_balanced_d,
                                    y_train=y_train_balanced_d,
                                    n_folds=10,
                                    random_state=123,
                                    stratified=True,
                                    shuffle=True
                                    )

    lm_x_train_probas = logistic_model.predict_proba(X_train_d)
    lm_x_test_probas = logistic_model.predict_proba(X_test_d)

    y_pred_full = logistic_model.predict(X_preprocessed_dummies)
    y_pred_test = logistic_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))

if classifiers['b_naive_bayes']:
    print('\n Training Bernoulli Naive Bayes \n')

    params_bnb = {'alpha': 1.0,
                  'binarize': 0.0,
                  'fit_prior': False
                  }

    b_naive_bayes_model = train_bernoulli_naivebayes(params_bnb,
                                                     x_train=X_train_balanced_d,
                                                     y_train=y_train_balanced_d,
                                                     n_folds=10,
                                                     random_state=123,
                                                     stratified=True,
                                                     shuffle=True
                                                     )

    bnn_x_train_probas = b_naive_bayes_model.predict_proba(X_train_d)
    bnn_x_test_probas = b_naive_bayes_model.predict_proba(X_test_d)

    y_pred_full = b_naive_bayes_model.predict(X_preprocessed_dummies)
    y_pred_test = b_naive_bayes_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))


if classifiers['random_forest']:
    print('\n Training Random Forest \n')

    params_rf = {'max_depth': 2,
                 'max_features': 5,
                 'min_samples_leaf': 2,
                 'min_samples_split': 2,
                 'n_estimators': 20,
                 'random_state': 123
                 }

    random_forest_model = train_random_forest(params_rf,
                                              x_train=X_train_balanced_d,
                                              y_train=y_train_balanced_d,
                                              n_folds=10,
                                              random_state=123,
                                              stratified=True,
                                              shuffle=True
                                              )

    rf_x_train_probas = random_forest_model.predict_proba(X_train_d)
    rf_x_test_probas = random_forest_model.predict_proba(X_test_d)

    y_pred_full = random_forest_model.predict(X_preprocessed_dummies)
    y_pred_test = random_forest_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))

if classifiers['svm']:
    print('\n Training SVM \n')

    params_svm = {'C': 0.01,
                  'class_weight': 'balanced',
                  'probability': True
                  }

    svm_model = train_svm(params_svm,
                          x_train=X_train_balanced_d,
                          y_train=y_train_balanced_d,
                          n_folds=10,
                          random_state=123,
                          stratified=True,
                          shuffle=True
                          )

    svm_x_train_probas = svm_model.predict_proba(X_train_d)
    svm_x_test_probas = svm_model.predict_proba(X_test_d)

    y_pred_full = svm_model.predict(X_preprocessed_dummies)
    y_pred_test = svm_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion matrix")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))


if classifiers['c_naive_bayes']:
    print('\n Training Complement Naive Bayes \n')

    params_cnb = {'alpha': 1.0,
                  'class_prior': None,
                  'fit_prior': True,
                  'norm': False
                  }

    cnb_model = train_complement_naivebayes(params_cnb,
                                            x_train=X_train_balanced_d,
                                            y_train=y_train_balanced_d,
                                            n_folds=10,
                                            random_state=123,
                                            stratified=True,
                                            shuffle=True
                                            )

    cnb_x_train_probas = cnb_model.predict_proba(X_train_d)
    cnb_x_test_probas = cnb_model.predict_proba(X_test_d)

    y_pred_full = cnb_model.predict(X_preprocessed_dummies)
    y_pred_test = cnb_model.predict(X_test_d)

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
    print(profit_score_function(y_test_d, y_pred_test))
    print("Confusion")
    confusion_matrix_report(y_test_d, y_pred_test)
    print("Acc")
    print(accuracy_score(y_test_d, y_pred_test))
    print("Precision")
    print(precision_score(y_test_d, y_pred_test))
    print("Recall")
    print(recall_score(y_test_d, y_pred_test))
    print("F1")
    print(f1_score(y_test_d, y_pred_test))

# Advanced Classifiers

if classifiers['xgboost']:
    print('\n Training XGBoost \n')

    xgb_params_1 = {
        'gamma': 0.05,
        'booster': 'gblinear',
        'max_depth': 3,
        'min_child_weight': 1,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_lambda': 0.01,
        'reg_alpha': 0,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'nthread': -1,
        'seed': 27
    }

    xgb_model_1 = train_xgb_model(params=xgb_params_1,
                                  x_train=X_train_balanced_l,
                                  y_train=y_train_balanced_l,
                                  n_folds=10,
                                  random_state=123
                                  )

    xgb_x_train_probas = xgb_model_1.predict_proba(X_train_l)
    xgb_x_test_probas = xgb_model_1.predict_proba(X_test_l)

    y_pred_full = xgb_model_1.predict(X_preprocessed_label)
    y_pred_test = xgb_model_1.predict(X_test_l)

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

######################################### 2ND LEVEL TRAINING ###########################################################

print('\n First level predictions finished. \n')

print('\n Saving predictions to file. \n')

# Create dataframe with predicted probabilites from each 1st level model
train_probas = pd.DataFrame()
test_probas = pd.DataFrame()

if classifiers['g_naive_bayes']:
    train_probas['g_naive_bayes'] = gnb_x_train_probas[:,1]
    test_probas['g_naive_bayes'] = gnb_x_test_probas[:,1]
if classifiers['nearest_centroid']:
    train_probas['nearest_centroid'] = nc_x_train_preds
    test_probas['nearest_centroid'] = nc_x_test_preds
if classifiers['knn']:
    train_probas['knn'] = knn_x_train_probas[:,1]
    test_probas['knn'] = knn_x_test_probas[:,1]
if classifiers['decision_tree']:
    train_probas['decision_tree'] = dt_x_train_probas[:,1]
    test_probas['decision_tree'] = dt_x_test_probas[:,1]
if classifiers['b_naive_bayes']:
    train_probas['b_naive_bayes'] = bnn_x_train_probas[:,1]
    test_probas['b_naive_bayes'] = bnn_x_test_probas[:,1]
if classifiers['logistic']:
    train_probas['logistic'] = lm_x_train_probas[:,1]
    test_probas['logistic'] = lm_x_test_probas[:,1]
if classifiers['random_forest']:
    train_probas['random_forest'] = rf_x_train_probas[:,1]
    test_probas['random_forest'] = rf_x_test_probas[:,1]
if classifiers['svm']:
    train_probas['svm'] = svm_x_train_probas[:,1]
    test_probas['svm'] = svm_x_test_probas[:,1]
if classifiers['c_naive_bayes']:
    train_probas['c_naive_bayes'] = cnb_x_train_probas[:,1]
    test_probas['c_naive_bayes'] = cnb_x_test_probas[:,1]
if classifiers['xgboost']:
    train_probas['xgboost'] = xgb_x_train_probas[:,1]
    test_probas['xgboost'] = xgb_x_test_probas[:,1]

# set index
train_probas = train_probas.set_index(y_train_l.index)
test_probas = test_probas.set_index(y_test_l.index)

# save to csvs for future reference
train_probas.to_csv('1st_level_probs_train.csv', index=None)
test_probas.to_csv('1st_level_probs_test.csv', index=None)

# reconstruct full dataset
x_full_reconstructed = train_probas.append(test_probas)
y_full_reconstructed = y_train_l.append(y_test_l)

# drop classifiers not used for ensemble
x_full_reconstructed = x_full_reconstructed.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)
train_probas = train_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)
test_probas = test_probas.drop(['c_naive_bayes', 'g_naive_bayes', 'b_naive_bayes', 'nearest_centroid', 'knn'], axis=1)

# balance the new train data
x_train_probas_balanced, y_train_probas_balanced = data_balancing(train_probas, y_train_l)

print('\n Starting with second level predictions. \n')

print('\n Training Ensemble using XGBoost \n')

ensemble_xgb_params_1 = {'booster': 'gblinear',
                         'colsample_bytree': 0.6,
                         'gamma': 0.05,
                         'learning_rate': 0.01,
                         'max_depth': 3,
                         'min_child_weight': 1,
                         'n_estimators': 1000,
                         'nthread': -1,
                         'objective': 'binary:logistic',
                         'reg_alpha': 0,
                         'reg_lambda': 1,
                         'seed': 27,
                         'subsample': 0.6}

ensemble_model_xgb_1 = train_xgb_ensemble(y_train=y_train_l,
                                          x_train=train_probas,
                                          y_train_b=y_train_probas_balanced,
                                          x_train_b=x_train_probas_balanced,
                                          y_test=y_test_l,
                                          x_test=test_probas,
                                          params=ensemble_xgb_params_1,
                                          n_folds=10,
                                          random_state=123
                                          )

print('Training Ensemble using RandomForest')

ensemble_params_rf = {
        'n_estimators': 15,
        'max_depth': 21,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 5}

ensemble_model_rf = train_rf_ensemble(x_train=train_probas,
                                      y_train=y_train_l,
                                      y_train_b=y_train_probas_balanced,
                                      x_train_b=x_train_probas_balanced,
                                      y_test=y_test_l,
                                      x_test=test_probas,
                                      params=ensemble_params_rf,
                                      n_folds=10,
                                      random_state=123
                                      )

