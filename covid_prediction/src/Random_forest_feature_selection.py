#%%
#importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

#%%
df=pd.read_csv('data/svm_output.csv', low_memory=False)
df.head()
df.info()

#%%
X=df.drop(['corona_result','model_prediction'] ,axis=1)
y=df['model_prediction']

#%%
model = RandomForestClassifier(n_estimators = 100)
model.fit(X, y)
#%%
importance=model.feature_importances_
print(importance)
for i,v in enumerate(importance):
    print('Feature: %s - score: %.5f'  %(df.columns[i],v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()
df['test_indication'].value_counts()