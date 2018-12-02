# Imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels

######################################### DISCLAIMER ###################################################################

# This class features all methods used for evaluating predictions and determining the best hyperparameters through
# grid search.

########################################################################################################################

__author__ = "Hendrik Roeder"

# Set a default value for k in cross validations
k = 10

# Set the values for profit calculation
profit_customer = 150 # Excluding the cost of a call!
cost_call = 10 # State a positive value!

profit_tp = profit_customer - cost_call # Currently 140
profit_fp = -cost_call # Currently -10


# Print a confusion matrix report (based on exercise 3)
def confusion_matrix_report(y_true, y_pred):
    cm, labels = confusion_matrix(y_true, y_pred), unique_labels(y_true, y_pred)
    column_width = max([len(str(x)) for x in labels] + [6])  # 5 is value length
    report = " " * column_width + " " + "{:_^{}}".format("Prediction", column_width * len(labels)) + "\n"
    report += " " * column_width + " ".join(["{:>{}}".format(label, column_width) for label in labels]) + "\n"
    for i, label1 in enumerate(labels):
        report += "{:>{}}".format(label1, column_width) + " ".join(
            ["{:{}d}".format(cm[i, j], column_width) for j in range(len(labels))]) + "\n"
    print(report)


# Grid search generalized (generalization of exercise 5)
def grid_search(model, features, target, positive_label, parameters, fit_params, score, folds):
    k = folds
    if score == "precision":
        model_scorer = make_scorer(precision_score, pos_label=positive_label)
        scoring = score
    elif score == "recall":
        model_scorer = make_scorer(recall_score, pos_label=positive_label)
        scoring = score
    elif score == "f1":
        model_scorer = make_scorer(f1_score, pos_label=positive_label)
        scoring = score
    else:
        model_scorer = make_scorer(accuracy_score)
        scoring = "accuracy"
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=model_scorer, verbose=5,
                                         cv=cross_validation, fit_params=fit_params)
    grid_search_estimator.fit(features, target)

    print("best " + scoring + " is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))


# Grid search that returns the best model
def grid_search_model(model, features, target, positive_label, parameters, fit_params, score, folds):
    k = folds
    if score == "precision":
        model_scorer = make_scorer(precision_score, pos_label=positive_label)
        scoring = score
    elif score == "recall":
        model_scorer = make_scorer(recall_score, pos_label=positive_label)
        scoring = score
    elif score == "f1":
        model_scorer = make_scorer(f1_score, pos_label=positive_label)
        scoring = score
    elif score == "roc_auc":
        model_scorer = "roc_auc"
        scoring = "roc_auc"
    else:
        model_scorer = make_scorer(accuracy_score)
        scoring = "accuracy"
    print('grid search started with ' + str(k) + ' folds')
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=model_scorer,
                                         cv=cross_validation, fit_params=fit_params, verbose=2, n_jobs= -1)
    grid_search_estimator.fit(features, target)

    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))

    print("best " + scoring + " is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))

    return grid_search_estimator.best_estimator_


# Print a cost matrix
def cost_matrix(y_true, y_pred, cost_fp, cost_fn):
    cm = confusion_matrix(y_true, y_pred)
    cost = cm[0][1] * cost_fp + cm[1][0] * cost_fn
    print('Total cost of the model: ' + str(cost))


# Print a profit matrix
def profit_matrix(y_true, y_pred, profit_tp, profit_fp):
    cm = confusion_matrix(y_true, y_pred)
    profit = cm[1][1] * profit_tp + cm[0][1] * profit_fp
    print('Total profit of the model: ' + str(profit))
    print(cm)


# Define the profit score function
def profit_score_function(y_true, y_predicted):
    score = 0
    score_i = 0
    for i, v in enumerate(y_true, 1):
        if y_predicted[i-1] == 1 and v == 1: # TP
            score_i = profit_tp
        if y_predicted[i-1] == 1 and v == 0: # FP (Called, but no subscription)
            score_i = profit_fp # profit_fp is negative!
        if y_predicted[i-1] == 0 and v == 1: # FN (Missed, but would have subscribed)
            score_i = 0 # Otherwise it's redundant
        if y_predicted[i-1] == 0 and v == 0:
            score_i = 0 # Otherwise it's redundant
        score += score_i
    return score


# Run a grid search based on the cost function and return the best model
def grid_search_cost_model(model, features, target, parameters, fit_params, folds):
    k = folds
    model_scorer = make_scorer(profit_score_function, greater_is_better=True)
    print('grid search started with ' + str(k) + ' folds')
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=model_scorer,
                                         cv=cross_validation, fit_params=fit_params, verbose=2)
    grid_search_estimator.fit(features, target)

    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))

    print("best profit is {} with params {} ".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))

    return grid_search_estimator.best_estimator_


# Function to get a scorer for the profit function
def get_profit_scorer ():
    return make_scorer(profit_score_function, greater_is_better=True)


# Run a grid search based on the cost function and return the best parameters
def grid_search_cost_params(model, features, target, parameters, fit_params, folds):
    k = folds
    print('grid search started with ' + str(k) + ' folds')
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=get_profit_scorer(),
                                         cv=cross_validation, fit_params=fit_params, verbose=2)
    grid_search_estimator.fit(features, target)

    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))

    print("best profit is {} with params {} ".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))

    return grid_search_estimator.best_params_