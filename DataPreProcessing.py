import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from category_encoders.one_hot import OneHotEncoder

__author__ = "Soumya Arun Barikeri"

######################################### DISCLAIMER ###################################################################

# This class features all methods used for preprocessing and feature engineering.

########################################################################################################################

# Check for missing values in the data set
def check_missing_values(data_set):
    if data_set.isnull().values.any():
        print('There are missing values in the dataset.')
    else:
        print('There are no missing values in the dataset.')

    columns = list(data_set.columns)

    # Fill mean value in case of numerical data, and mode in case of categorical
    for column in columns:
        if data_set[column].isnull().values.any():
            print('There are missing values in the column ' + column)
            if data_set[column].select_dtypes(include=[np.number]):
                mean_value = data_set[column].mean()
                print('Filling the missing value with mean value of the column ' + column)
                data_set[column] = data_set[column].fillna(mean_value)
            else:
                print('Filling the missing value with mode value of the column ' + column)
                data_set[column] = data_set[column].fillna(data_set[column].mode().iloc[0])


def data_preprocessing(data_set, columns_to_drop, columns_to_onehot, columns_to_dummy, columns_to_label, normalise):
    # Drop the specified columns
    if len(columns_to_drop) != 0:
        data_set_dropped_columns = data_set.drop(columns_to_drop, axis=1)
    else:
        data_set_dropped_columns = data_set

    # Extract data set with numeric values
    bank_data_num = data_set_dropped_columns.select_dtypes(include=[np.number])
    # Get the columns with numeric(continuous) values
    numeric_columns = list(bank_data_num.columns.values)

    columns_needed = columns_to_onehot + columns_to_dummy + columns_to_label + numeric_columns
    # Get the data set which is not specified
    data_left = data_set_dropped_columns.drop(columns_needed, axis=1)

    # Get subsets of data according to encoder to be applied
    data_to_onehot_encode = data_set_dropped_columns[columns_to_onehot]
    data_to_dummy_encode = data_set_dropped_columns[columns_to_dummy]
    data_to_label_encode = data_set_dropped_columns[columns_to_label]

    # Normalising data using Min-Max scaling => [(x - min(x)) / (min(x) - max(x))]
    if normalise:
        x = data_set[numeric_columns].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        bank_data_no_cols = pd.DataFrame(x_scaled, columns=numeric_columns, index=data_set.index)
        data_set[numeric_columns] = bank_data_no_cols
    bank_data_num_norm = data_set[numeric_columns]

    # One-hot encoding specified columns
    if len(columns_to_onehot) != 0:
        encoder = OneHotEncoder()
        bank_data_onehotencoded = encoder.fit_transform(data_to_onehot_encode)
    else:
        bank_data_onehotencoded = pd.DataFrame(np.zeros((data_set.shape[0], 1)))

    # Encoding Categorical variables - Creating k dummy variables for an original variable k-categories
    if len(columns_to_dummy) != 0:
        bank_data_dummyencoded = pd.get_dummies(data_to_dummy_encode)
    else:
        bank_data_dummyencoded = pd.DataFrame(np.zeros((data_set.shape[0], 1)))

    # Label encoding specified columns
    if len(columns_to_label) != 0:
        for i in data_to_label_encode.columns.values:
            lbl = preprocessing.LabelEncoder()
            data_to_label_encode[i] = lbl.fit_transform(data_to_label_encode[i])
    else:
        data_to_label_encode = pd.DataFrame(np.zeros((data_set.shape[0], 1)))

    # Dummy encode the unspecified data columns by default
    if not data_left.empty:
        print('in if not statement')
        bank_data_leftencoded = pd.get_dummies(data_left)
    else:
        bank_data_leftencoded = pd.DataFrame(np.zeros((data_set.shape[0], 1)))

    print('Data left out is...')
    print(bank_data_leftencoded.shape)
    print(bank_data_leftencoded.head())
    print('Data one-hot encoded is...')
    print(bank_data_onehotencoded.shape)
    print(bank_data_onehotencoded.head())
    print('Data dummy encoded is...')
    print(bank_data_dummyencoded.shape)
    print(bank_data_dummyencoded.head())
    print('Data label encoded is...')
    print(data_to_label_encode.shape)
    print(data_to_label_encode.head())
    print('Numerical Data normalised is...')
    print(bank_data_num_norm.shape)
    print(bank_data_num_norm.head())

    bank_data_norm_encoded = np.concatenate((bank_data_onehotencoded, bank_data_dummyencoded,data_to_label_encode,bank_data_num_norm, bank_data_leftencoded), axis=1)
    bank_data_norm_encoded = pd.DataFrame(bank_data_norm_encoded)
    print('Data after pre-processing')
    print(bank_data_norm_encoded.head())

    # Check if all categorical columns are encoded
    print('Checking datatypes..')
    tmp = 0
    for i in bank_data_norm_encoded.columns.values:
        if bank_data_norm_encoded[i].dtype == object:
            tmp = tmp + 1
    if tmp == 0:
        print('All columns are encoded.')
    else:
        print('Not all columns are encoded')

    return bank_data_norm_encoded


# New feature if age is greater than 60
def elder_person(data_set):
    data_set['elder'] = np.where(data_set['age'] > 60, 1, 0)
    return data_set


# New feature called student
def is_student(data_set):
    data_set['student'] = np.where(data_set['job'] == "student", 1, 0)
    return data_set


# New feature if mode of contact is cellular
def cellular_contact(data_set):
    data_set['cellular'] = np.where(data_set['contact'] == "cellular", 1, 0)
    return data_set


# New feature for euribor3m in range [1.5,2.5)
def euribor_bin(data_set):
    data_set["euribor_bin"] = 0
    for index, row in data_set.iterrows():
        if row["euribor3m"] >= 1.5 and row["euribor3m"] < 2.5:
            data_set["euribor_bin"] = 1
    return data_set


# New feature for student with university degree
def in_education(data_set):
    data_set["in_education"] = 0
    for index, row in data_set.iterrows():
        count = 0
        if row["age"] < 18:
            count += 1
        if row["job"] == "student":
            count += 1
        if row["education"] == "university.degree":
            count += 1
        if count >= 2:
            row["in_education"] = 1
    return data_set


# Binning age feature into 4 bins
def bin_age(data_set):
    bins = [0, 17, 34, 60, 100]
    data_set['age'] = pd.cut(data_set['age'], bins, labels=['Child', 'Adult', 'Middle_aged', 'Old'])
    return data_set['age'].astype('object')


# Binning duration
def bin_duration(data_set):
    duration_in_min = data_set['duration']/60
    bins = [0, 5, 10, 90]
    data_set['duration'] = pd.cut(duration_in_min, bins, labels=['lessThan5min', '5minTo10min', 'moreThan10min'])
    return data_set['duration'].astype('object')


# New feature if customer not contacted earlier
def not_contacted(data_set):
    data_set['not_contacted'] = 0
    data_set.loc[data_set['pdays'] == 999, 'not_contacted'] = 1
    return data_set


# New feature if customer was contacted in last 9 months
def contacted_last_9_months(data_set):
    data_set['contacted_last_9_months'] = 0
    data_set.loc[data_set['pdays'] < 10, 'contacted_last_9_months'] = 1
    return data_set

# New feature if number of calls is less than 20
def campaign_split(data_set):
    data_set['campaign_many_calls'] = 0
    data_set.loc[data_set['campaign'] < 20, 'campaign_many_calls'] = 1
    return data_set


# Data balancing for 1:1(yes:no) split
def data_balancing(x_train, y_train):
    y_train = pd.DataFrame(data=y_train)
    train_full_balance = pd.DataFrame(data=x_train)
    train_x_balance = train_full_balance[y_train["y"] == 1]
    train_x_balance_0 = train_full_balance[y_train["y"] == 0]
    count_pos = train_x_balance.shape[0]
    train_xy_balance_0 = train_x_balance_0.assign(y=0)
    train_xy_balance = train_x_balance.assign(y=1)
    train_xy_balance_0_sample = train_xy_balance_0.sample(n=count_pos, replace=False, random_state=42)
    train_full_balance = train_xy_balance.append(train_xy_balance_0_sample)
    x_train = train_full_balance.drop(['y'], axis=1)
    y_train = train_full_balance['y']
    return x_train, y_train
