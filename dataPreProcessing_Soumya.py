import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from category_encoders.one_hot import OneHotEncoder


#Check for missing values in the data set and fill it with mean value of the column
def check_missing_values(data_set):
    if data_set.isnull().values.any() == True:
        print('There are missing values in the dataset.')
    else:
        print('There are no missing values in the dataset.')

    columns = list(data_set.columns)

    for column in columns:
        if data_set[column].isnull().values.any() == True:
            print('There are missing values in the column ' + column)
            mean_value = data_set[column].mean()
            print('Filling the missing value with mean value of the column ' + column)
            data_set[column] = data_set[column].fillna(mean_value)



def data_preprocessing(data_set, columns_to_drop, columns_to_onehot, columns_to_dummy, columns_to_label, normalise):
    data_set_dropped_columns = data_set.drop(columns_to_drop, axis=1)
#    bank_data_new = data_set_dropped_columns[columns_to_preprocess]
    data_to_onehot_encode = data_set_dropped_columns[columns_to_onehot]
    data_to_dummy_encode = data_set_dropped_columns[columns_to_dummy]
    data_to_label_encode = data_set_dropped_columns[columns_to_label]
    data_set.info(memory_usage='deep')
    print(data_set.describe(include=[np.number]))

    #Extact columns with numeric values
    bank_data_num = data_set_dropped_columns.select_dtypes(include=[np.number])
    #Get the columns with numeric(continuous) values
    numeric_columns = list(bank_data_num.columns.values)
    print(numeric_columns)

    #Normalising data using Min-Max scaling => [(x - min(x)) / (min(x) - max(x))]
    if normalise:
        x = data_set[numeric_columns].values  #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        bank_data_no_cols = pd.DataFrame(x_scaled, columns=numeric_columns, index=data_set.index)
        data_set[numeric_columns] = bank_data_no_cols
    bank_data_num_norm = data_set[numeric_columns]
    # Encoding Categorical variables - Creating k dummy variables for an original variable k-categories
#    bank_data_norm = bank_data_new
    bank_data_onehotencoded = []
    bank_data_dummyencoded = []
    bank_data_labelencoded = []
    if len(columns_to_onehot) != 0:
#        bank_data_cat = bank_data_new.select_dtypes(include=['object'])
        # Get the columns with categorical values
#        cat_columns = list(bank_data_cat.columns.values)
#        print(cat_columns)
#        to_be_encoded = cat_columns
        encoder = OneHotEncoder()
        bank_data_onehotencoded = encoder.fit_transform(data_to_onehot_encode)

        # Concatenate encoded attributes with continuous attributes
#        bank_data_numerical = bank_data_new.drop(columns_to_onehot, axis=1)
#        bank_data_norm_encoded = np.concatenate((bank_data_encoded, bank_data_numerical), axis=1)
#
#        bank_data_norm_encoded = pd.DataFrame(bank_data_norm_encoded)
        # print(bank_data_norm_encoded.head())
    if len(columns_to_dummy) != 0:
        bank_data_dummyencoded = pd.get_dummies(data_to_dummy_encode)
    if len(columns_to_label) != 0:
        for i in data_to_label_encode.columns.values:
            lbl = preprocessing.LabelEncoder()
            data_to_label_encode[i] = lbl.fit_transform(data_to_label_encode[i])

    print(bank_data_onehotencoded.shape)
    print(bank_data_onehotencoded.head())
    print(bank_data_dummyencoded.shape)
    print(bank_data_dummyencoded.head())
    print(data_to_label_encode.shape)
    print(data_to_label_encode.head())
    print(bank_data_num_norm.shape)
    print(bank_data_num_norm.head())

    bank_data_norm_encoded = np.concatenate((bank_data_onehotencoded, bank_data_dummyencoded,data_to_label_encode,bank_data_num_norm), axis=1)
    bank_data_norm_encoded = pd.DataFrame(bank_data_norm_encoded)
    print('Data after pre-processing')
    print(bank_data_norm_encoded.head())

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


def data_binned(data_set, columns_to_bin):
    bank_data_binned = []
    for columns in columns_to_bin:
        feature = pd.cut(data_set[columns], bins=4, labels=['one', 'two','three','four'])
        bank_data_binned.append(feature)

    binned_data_2d = np.column_stack(bank_data_binned)
    print(binned_data_2d)
    print(binned_data_2d.shape)
    return binned_data_2d


def bin_age(data_set):
    bins = [0, 17, 34, 60, 100]
    data_set['age'] = pd.cut(data_set['age'], bins, labels=['Child', 'Adult', 'Middle_aged', 'Old'])
#    print(data_set['age'])
    return data_set['age']


def bin_duration(data_set):
    duration_in_min = data_set['duration']/60
    bins = [0, 5, 10, 90]
    data_set['duration'] = pd.cut(duration_in_min, bins, labels=['lessThan5min', '5minTo10min', 'moreThan10min'])
#    print(data_set['duration'])
    return data_set['duration']


def not_contacted(data_set):
   data_set['not_contacted'] = 0
   data_set.loc[data_set['pdays'] == 999 ,'not_contacted'] = 1
   return data_set


def bin_pdays(data_set):
   data_set.loc[data_set['pdays'] == 999 ,'pdays'] = 0
   pmonths = data_set['pdays'] / 30
   bins = [0, 3, 7, 1000]
   data_set['pmonths'] = pd.cut(pmonths, bins, labels=['0To3months', '3To7months', 'moreThan7months'])
   return data_set

def data_balancing(X_train, y_train):
    y_train = pd.DataFrame(data=y_train)
    train_full_balance= pd.DataFrame(data=X_train)
    train_x_balance = train_full_balance[y_train["y"]== 1]
    train_x_balance_0 = train_full_balance[y_train["y"]== 0]
    count_pos = train_x_balance.shape[0]
    train_xy_balance_0 = train_x_balance_0.assign(y=0)
    train_xy_balance = train_x_balance.assign(y=1)
    train_xy_balance_0_sample=train_xy_balance_0.sample(n=count_pos, replace=False, random_state=42)
    train_full_balance = train_xy_balance.append(train_xy_balance_0_sample)
    X_train = train_full_balance.drop(['y'], axis=1)
    y_train = train_full_balance['y']
    return X_train, y_train

bank_data = pd.read_csv('/Users/Soumya/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')
print(pd.__version__)
print(bank_data.head())
to_be_preprocessed = ['age','duration','campaign','pdays','emp.var.rate','cons.price.idx','job','marital','education','default','housing','loan','contact','month','day_of_week','previous']
columns_to_bin = ['age','duration','campaign','pdays','emp.var.rate','campaign']
columns_to_drop = ['poutcome']
columns_to_onehot = ['marital', 'education', 'default', 'housing']
columns_to_dummy = ['loan', 'contact', 'day_of_week']
columns_to_label = ['job', 'month']
check_missing_values(bank_data)

binned_age = bin_age(bank_data)
bank_data['age'] = binned_age.astype('object')
binned_duration = bin_duration(bank_data)
bank_data['duration'] = binned_duration.astype('object')
print(bank_data.head())

#binned_data = data_binned(bank_data, columns_to_bin)
#data_frame_binned = pd.DataFrame(binned_data)
#print(data_frame_binned.head())
#frames = [bank_data, data_frame_binned]
#result_data = pd.concat(frames)
data_preprocessed = data_preprocessing(bank_data, columns_to_drop, columns_to_onehot, columns_to_dummy, columns_to_label, True)

