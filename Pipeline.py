# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import modelTrainingMarius as modelTraining

### Data Loading #####

#Print all rows and columns. Dont hide any.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

bankingcalldata = pd.read_csv('/Users/mariusbock/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')

#### This is just for the model guys to train test their models #######

print('Full dataset shape: ')
print(bankingcalldata.shape)

from datetime import datetime
from sklearn.metrics import mean_absolute_error, accuracy_score, average_precision_score, recall_score, confusion_matrix
from scipy.stats import skew, boxcox
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import itertools

if bankingcalldata.isnull().values.any() == True:
    print('There are missing values in the dataset.')
else:
    print('There are no missing values in the dataset.')

columns = list(bankingcalldata.columns)

for column in columns:
    if bankingcalldata[column].isnull().values.any() == True:
        print('There are missing values in the column ' + column)

# Variable to hold the list of variables for an attribute in the train and test data
labels = []
to_be_encoded = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                 'previous','poutcome']

print('Encoding categorical columns..')

for i in bankingcalldata.columns.values:
    if bankingcalldata[i].dtype == object:
        lbl = preprocessing.LabelEncoder()
        bankingcalldata[i] = lbl.fit_transform(bankingcalldata[i])

for i in range(len(to_be_encoded)):
    labels.append(list(bankingcalldata[to_be_encoded[i]].unique()))

# One hot encode all categorical attributes
cats = []
encoded_data = bankingcalldata.drop('y', axis=1)

print('Checking datatypes..')
tmp = 0
for i in bankingcalldata.columns.values:
    if bankingcalldata[i].dtype == object:
        tmp = tmp + 1
if tmp == 0:
    print('All columns are encoded.')
else:
    print('Not all columns are encoded')

print('Finished.')

X_full = bankingcalldata.drop('y', axis=1)
y_full = bankingcalldata['y']
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.40, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_full.value_counts())

#### END ##############

xgb_params_1 = {
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'nthread': -1,
        'scale_pos_weight': 1,
        'seed': 27,
        'eval_metric': 'logloss',
        #'num_class': 1,
        'silent': 1
    }

xgb_model_1 = modelTraining.train_xgb_model(xgb_params_1,
                                            X_train,
                                            y_train,
                                            rounds=10000,
                                            early_stopping=50,
                                            n_folds=10,
                                            random_state=123)


