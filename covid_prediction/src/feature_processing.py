#%%
#importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
print("yes")
#%%
#Loading the covid dataset

df=pd.read_csv('data/corona_tested_individuals_ver_006.english.csv', low_memory=False)
df.head()
df.info()

#%%
#Converting to Datetime dtype
df['test_date']=pd.to_datetime(df['test_date'])
df['date']=df['test_date'].dt.day
df['month']=df['test_date'].dt.month
df.drop('test_date',axis=1,inplace=True)

#%%
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
#%%
#processing the sore_throat column
df['sore_throat_not_none']=df['sore_throat'].apply(lambda val:0 if val=='None' else 1)
df['sore_throat']=labelencoder.fit_transform(df['sore_throat'])

#%%
#processing the cough column
df['cough_not_none']=df['cough'].apply(lambda val:0 if val=='None' else 1)
df['cough']=labelencoder.fit_transform(df['cough'])

#%%
#processing the fever column
df['fever_not_none']=df['fever'].apply(lambda val:0 if val=='None' else 1)
df['fever']=labelencoder.fit_transform(df['fever'])


#%%
#converting the object type Shortness of breath and head ache to int
df['shortness_of_breath_not_none']=df['shortness_of_breath'].apply(lambda val:0 if val=='None' else 1)
df['shortness_of_breath']=labelencoder.fit_transform(df['shortness_of_breath'])

df['head_ache_not_none']=df['head_ache'].apply(lambda val:0 if val=='None' else 1)
df['head_ache']=labelencoder.fit_transform(df['head_ache'])

#%%
#processing the string categorical columns

#Processing gender column

df['gender_present']=df['gender'].apply(lambda val:0 if val=='None' else 1)
df['gender']=labelencoder.fit_transform(df['gender'])

#Processing age_60_and_above column
df['age_present']=df['age_60_and_above'].apply(lambda val:0 if val=='None' else 1)
df['age_60_and_above']=labelencoder.fit_transform(df['age_60_and_above'])

#Processing test_indication column
df['test_indication']=labelencoder.fit_transform(df['test_indication'])
df.info()

#there are 3 unique ategories in corona_resul


df.to_csv('data/processed_data.csv', index=False)
df.corona_result.value_counts()