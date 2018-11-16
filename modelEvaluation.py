# Imports
import numpy as np
#import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels

# Add loss functions -> check wikipedia article
# Check if grid search can output ALL measures together


# Set value for k in cross validations
k = 10


# Confusion Matrix Report for prediction results (from Exercise 3)
def confusion_matrix_report(y_true, y_pred):
    cm, labels = confusion_matrix(y_true, y_pred), unique_labels(y_true, y_pred)
    column_width = max([len(str(x)) for x in labels] + [6])  # 5 is value length
    report = " " * column_width + " " + "{:_^{}}".format("Prediction", column_width * len(labels)) + "\n"
    report += " " * column_width + " ".join(["{:>{}}".format(label, column_width) for label in labels]) + "\n"
    for i, label1 in enumerate(labels):
        report += "{:>{}}".format(label1, column_width) + " ".join(
            ["{:{}d}".format(cm[i, j], column_width) for j in range(len(labels))]) + "\n"
    print(report)


# Run a confusion matrix report based on a cross validation
def confusion_matrix_report_cv(model, features, target):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    predictions = cross_val_predict(model, features, target, cv=cross_validation)
    confusion_matrix_report(target, predictions)


# Cross validation for accuracy
def cv_accuracy(model, features, target):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    accuracies = cross_val_score(model, features, target, cv=cross_validation, scoring='accuracy')
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    avg_accuracy = np.mean(accuracies)

    print("Minimum accuracy: " + min_accuracy)
    print("Maximum accuracy: " + max_accuracy)
    print("Average accuracy: " + avg_accuracy)


def classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


# Print a classification report based on a cross validation
def classification_report_cv(model, features, target):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    predictions = cross_val_predict(model, features, target, cv=cross_validation)
    print(classification_report(target, predictions))


# # Print ROC curve (based on exercise 4)
# def print_roc(models, features, target, positive_label):
#     plt.figure(figsize=(10, 10))
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)  # draw diagonal
#
#     count = 1
#     for model in models:
#         mean_fpr, mean_tpr, mean_auc, std_auc = get_roc(k, model, features.values, target, positive_label)
#         plt.plot(mean_fpr, mean_tpr, label='Model '+count+' (AUC: {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc))
#         count += 1
#
#     plt.xlabel('false positive rate')
#     plt.ylabel('true positive rate')
#     plt.legend()
#     plt.show()


# Grid search generalized (generalization of exercise 5)
def grid_search(model, features, target, positive_label, parameters, fit_params, score):
    if (score == "precision"):
        scoring = "precision_score"
    elif (score == "recall"):
        scoring = "recall_score"
    elif (score == "f1"):
        scoring = "f1_score"
    else:
        scoring = "accuracy_score"
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    model_scorer = make_scorer(scoring, pos_label=positive_label)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=model_scorer,
                                         cv=cross_validation, verbose=50)
    grid_search_estimator.fit(features, target)

    print("best" + scoring + " is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))


# Grid search for accuracy (based on exercise 5)
def grid_search_accuracy(model, features, target, positive_label, parameters):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    accuracy_scorer = make_scorer(accuracy_score, pos_label=positive_label)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=accuracy_scorer, cv=cross_validation)
    grid_search_estimator.fit(features, target)

    print("best accuracy is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))


# Grid search for precision (based on exercise 5)
def grid_search_precision(model, features, target, positive_label, parameters):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    precision_scorer = make_scorer(precision_score, pos_label=positive_label)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=precision_scorer, cv=cross_validation)
    grid_search_estimator.fit(features, target)

    print("best f1-score is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))


# Grid search for recall (based on exercise 5)
def grid_search_recall(model, features, target, positive_label, parameters):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    recall_scorer = make_scorer(recall_score, pos_label=positive_label)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=recall_scorer, cv=cross_validation)
    grid_search_estimator.fit(features, target)

    print("best f1-score is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))


# Grid search for f1 (based on exercise 5)
def grid_search_f1(model, features, target, positive_label, parameters):
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    f1_scorer = make_scorer(f1_score, pos_label=positive_label)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=f1_scorer, cv=cross_validation)
    grid_search_estimator.fit(features, target)

    print("best f1-score is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    results = grid_search_estimator.cv_results_
    for i in range(len(results['params'])):
        print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))


# Print a cost matrix (check order!)
def cost_matrix(y_true, y_pred, cost_fp, cost_fn):
    cm = confusion_matrix(y_true, y_pred)
    cost = cm[0][1] * cost_fp + cm[1][0] * cost_fn
    print('Total cost of the model: ' + cost)


# Calculate roc_avg (from exercise 4)
def get_roc(model, features, target, positive_label):
    mean_fpr = np.linspace(0, 1, 100)  # = [0.0, 0.01, 0.02, 0.03, ... , 0.99, 1.0]
    tprs = []
    aucs = []
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    for train_indices, test_indices in cross_validation.split(features, target):
        train_data = features[train_indices]
        train_target = target[train_indices]
        model.fit(train_data, train_target)

        test_data = features[test_indices]
        test_target = target[test_indices]
        decision_for_each_class = model.predict_proba(test_data)  # have to use predict_proba or decision_function

        fpr, tpr, thresholds = roc_curve(test_target, decision_for_each_class[:, 1], pos_label=positive_label)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  # tprs[-1] access the last element
        aucs.append(auc(fpr, tpr))

    # plt.plot(fpr, tpr)# plot for each fold

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # set the last tpr to 1
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, mean_auc, std_auc

def grid_search_model(model, features, target, positive_label, parameters, fit_params, score, listResults):
    if (score == "precision"):
        scoring = "precision_score"
    elif (score == "recall"):
        scoring = "recall_score"
    elif (score == "f1"):
        scoring = "f1_score"
    else:
        scoring = "accuracy_score"
    cross_validation = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    model_scorer = make_scorer(scoring, pos_label=positive_label)
    grid_search_estimator = GridSearchCV(model, parameters, scoring=model_scorer,
                                         cv=cross_validation)
    grid_search_estimator.fit(features, target, fit_params= fit_params)

    print("best" + scoring + " is {} with params {}".format(grid_search_estimator.best_score_,
                                                      grid_search_estimator.best_params_))
    if listResults == True:
        results = grid_search_estimator.cv_results_
        for i in range(len(results['params'])):
            print("{}, {}".format(results['params'][i], results['mean_test_score'][i]))

    return grid_search_estimator.best_estimator_

