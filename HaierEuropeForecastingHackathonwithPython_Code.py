# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:48:04 2024

@author: Rabia KAŞIKCI
"""
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

data= pd.read_csv("train.csv")
submission= pd.read_csv("sample_submission_joined.csv", delimiter=",")

#submission dosyasını düzenleyelim

submission=submission.drop(columns=['QTY'])
s_data_submission = submission[submission['Id'].str.startswith('S')]
p_data_submission = submission[submission['Id'].str.startswith('P')]



# Create a copy of the DataFrame
s_data_submission_copy = s_data_submission.copy()
p_data_submission_copy = p_data_submission.copy()

# Perform splitting and assignment on the copied DataFrames
s_data_submission_copy[['PH1', 'AREA_HYERARCHY2', 'MONTH']] = s_data_submission_copy['Id'].str.split(',', expand=True)
p_data_submission_copy[['PRODUCT', 'AREA_HYERARCHY1', 'WEEK']] = p_data_submission_copy['Id'].str.split(',', expand=True)


s_data_submission_copy['MONTH'] = s_data_submission_copy['MONTH'].str.replace('1', '10')
s_data_submission_copy['MONTH'] = s_data_submission_copy['MONTH'].str.replace('2', '11')
s_data_submission_copy['MONTH'] = s_data_submission_copy['MONTH'].str.replace('3', '12')
# Yeni değerlerin yer aldığı bir sözlük oluşturun
p_data_submission_copy['WEEK'] = p_data_submission_copy['WEEK'].astype(str).apply(lambda x: str(int(x) + 40) if int(x) <= 12 else x)


s_data_submission_copy=s_data_submission_copy.drop(columns=['Id'])
p_data_submission_copy=p_data_submission_copy.drop(columns=['Id'])

merged_submission = pd.concat([p_data_submission_copy, s_data_submission_copy], axis=1)



def week_number(date):
    return pd.to_datetime(date).isocalendar()[1]

# Hafta numaralarını içeren yeni bir sütun oluştur
data['WEEK'] = data['DATE'].apply(week_number)

merged_submission = pd.concat([p_data_submission_copy, s_data_submission_copy], axis=1)

unique_products = p_data_submission_copy['PRODUCT'].unique()

# Her bir unique ürün için ilgili verileri General Data'dan çek
all_product_data = pd.DataFrame()  # Tüm ürün verilerini depolamak için boş bir DataFrame oluşturulur
for product in unique_products:
    product_data = data[data['PRODUCT'] == product]
    all_product_data = pd.concat([all_product_data, product_data], ignore_index=True)

unique_products = s_data_submission_copy['PH1'].unique()
all_product_data_s= pd.DataFrame()  # Tüm ürün verilerini depolamak için boş bir DataFrame oluşturulur
for product in unique_products:
    product_data = data[data['PH1'] == product]
    all_product_data_s = pd.concat([all_product_data, product_data], ignore_index=True)

merged_train = pd.concat([all_product_data, all_product_data_s], axis=0)
merged_train = merged_train.drop_duplicates(subset=merged_train.columns, keep='first')


merged_train.to_csv('merged_train.csv', index=False)
train_yeni = merged_train.drop(columns=['DATE','YEAR','YEAR_QUARTER'])

x_train=train_yeni[train_yeni.columns[1:14]]
y_train=train_yeni[train_yeni.columns[0:1]]


new_column_names = ['AREA_HYERARCHY3', 'AREA_HYERARCHY4','PH2', 'PH3' ,'PH4', 'BRAND', 'SOURCE']
for column_name in new_column_names:
    merged_submission[column_name] = np.nan

merged_submission_df = pd.DataFrame(merged_submission)
data_df = pd.DataFrame(x_train)

# NaN değerleri doldurun
for column in merged_submission_df.columns:
    merged_submission_df[column].fillna(x_train[column].iloc[0], inplace=True)


#Feature Selection

merged_trainandsub = pd.concat([x_train, merged_submission_df], axis=0)

from sklearn.preprocessing import LabelEncoder

# Assuming cat_cols is a list of categorical column names
cat_cols = ['PRODUCT','PH1','BRAND','SOURCE','PH4']  # List your categorical column names here

# Perform label encoding
label_encoder = LabelEncoder()
for col in cat_cols:
    merged_trainandsub[col] = label_encoder.fit_transform(merged_trainandsub[col])


x_train_yeni=merged_trainandsub[:-67365]
y_train_yeni=y_train
x_test=merged_trainandsub.tail(67365).copy()


"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

# Mutual information tabanlı seçim
selector = SelectKBest(mutual_info_classif, k=8)  # En iyi 10 özellik
X_new = selector.fit_transform(x_train_yeni, y_train_yeni)  
selected_features = selector.get_support()
selected_feature_names = x_train_yeni.columns[selected_features]

print("Seçilen özelliklerin isimleri:")
print(selected_feature_names)
X_new_df = pd.DataFrame(data=X_new, columns=selected_feature_names)


removed_feature_names = [col for col in x_train_yeni.columns if col not in selected_feature_names]
x_test_yeni = x_test.drop(columns=removed_feature_names)
"""

#Modeller
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train_yeni, y_train_yeni)



# Tahmini y değerini hesaplayın
y_tahmin = model.predict(x_test)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train_yeni, y_train_yeni)
y_pred = rf_model.predict(x_test)

"""
from sklearn.neighbors import KNeighborsRegressor


knn_model = KNeighborsRegressor(n_neighbors=12)  # K = 5 için
knn_model.fit(x_train_yeni, y_train_yeni)
y_pred_knn= knn_model.predict(x_test)




from sklearn.svm import SVR

linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train_yeni, y_train_yeni)

# RBF çekirdeği ile SVR modelini oluşturun ve eğitin
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train_yeni, y_train_yeni)

# Modelleri test verisi üzerinde değerlendirin
linear_pred = linear_svr.predict(x_test)
rbf_pred = rbf_svr.predict(x_test)
"""