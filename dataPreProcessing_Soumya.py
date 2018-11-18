import pandas as pd
import numpy as np

bank_data = pd.read_csv('/Users/soumya/PycharmProjects/DMTeam20_Uni_Mannheim/input/bank-additional-full.csv', sep=';')
print(pd.__version__)
print(bank_data.head())
bank_data.info(memory_usage='deep')
print(bank_data.describe(include=[np.number]))

bank_data_target = np.array(bank_data['y'])
print(bank_data_target[:10])
#Extact columns with numeric values
bank_data_num = bank_data.select_dtypes(include=[np.number])
#Get the columns with numeric(continuous) values
numeric_columns = list(bank_data_num.columns.values)
numeric_columns

#Normalising data using Min-Max scaling => [(x - min(x)) / (min(x) - max(x))]
from sklearn import preprocessing

x = bank_data[numeric_columns].values  #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
bank_data_no_cols = pd.DataFrame(x_scaled, columns=numeric_columns, index=bank_data.index)
bank_data[numeric_columns] = bank_data_no_cols

#Encoding Categorical variables - Creating k dummy variables for an original variable k-categories
bank_data_norm = bank_data

bank_data_norm_encoded = pd.get_dummies(bank_data_norm.drop(columns='y'))
bank_data_norm_encoded.head()
