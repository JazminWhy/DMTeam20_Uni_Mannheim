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



def data_preprocessing(data_set,columns_to_preprocess,ONE_HOT):
    bank_data_new = data_set[columns_to_preprocess]
    bank_data_new.info(memory_usage='deep')
    print(bank_data_new.describe(include=[np.number]))

    #Extact columns with numeric values
    bank_data_num = bank_data_new.select_dtypes(include=[np.number])
    #Get the columns with numeric(continuous) values
    numeric_columns = list(bank_data_num.columns.values)
    print(numeric_columns)

    #Normalising data using Min-Max scaling => [(x - min(x)) / (min(x) - max(x))]
    x = bank_data_new[numeric_columns].values  #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    bank_data_no_cols = pd.DataFrame(x_scaled, columns=numeric_columns, index=bank_data_new.index)
    bank_data_new[numeric_columns] = bank_data_no_cols

    # Encoding Categorical variables - Creating k dummy variables for an original variable k-categories
    bank_data_norm = bank_data_new

    if ONE_HOT:
        bank_data_cat = bank_data_new.select_dtypes(include='object')
        # Get the columns with categorical values
        cat_columns = list(bank_data_cat.columns.values)
        print(cat_columns)
        to_be_encoded = cat_columns


        encoder = OneHotEncoder()

        bank_data_encoded = encoder.fit_transform(bank_data_norm[to_be_encoded])

        # Concatenate encoded attributes with continuous attributes
        bank_data_numerical = bank_data_new.drop(to_be_encoded, axis=1)
        bank_data_norm_encoded = np.concatenate((bank_data_encoded, bank_data_numerical), axis=1)

        bank_data_norm_encoded = pd.DataFrame(bank_data_norm_encoded)
        # print(bank_data_norm_encoded.head())
    else:
        if 'y' in columns_to_preprocess:
            bank_data_norm_encoded = pd.get_dummies(bank_data_norm.drop(columns='y'))
        else:
            bank_data_norm_encoded = pd.get_dummies(bank_data_norm)

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


bank_data = pd.read_csv('/Users/soumya/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')
print(pd.__version__)
print(bank_data.head())
to_be_preprocessed = ['age','duration','campaign','pdays','emp.var.rate','cons.price.idx','job','marital','education','default','housing','loan','contact','month','day_of_week','previous',
                 'poutcome']
columns_to_bin = ['age','duration','campaign','pdays','emp.var.rate','campaign']
check_missing_values(bank_data)
binned_data = data_binned(bank_data, columns_to_bin)
data_frame_binned = pd.DataFrame(binned_data)
print(data_frame_binned.head())
#frames = [bank_data, data_frame_binned]
#result_data = pd.concat(frames)
data_preprocessed = data_preprocessing(bank_data, to_be_preprocessed, True)
