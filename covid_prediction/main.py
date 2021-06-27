#%%
import pandas as pd
from src import config
df1=pd.read_csv('data/processed_data.csv')
df1.head()
df1.info()

#%%
'''
X=df1.drop('positive',axis=1)
y=df1['positive']
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

#%%
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
print("SVC complete.....")
#%%
predict=model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))'''
#%%
X=df1.drop('corona_result',axis=1)
df1.info()
df1['head_ache'].value_counts()
from sklearn.svm import OneClassSVM
svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=14729/(14729+260227+3892))
svm.fit(X)
predict=svm.predict(X)
df1['model_prediction']=predict

df1.info()
df1.model_prediction.value_counts()
df1['model_prediction'].value_counts()

df1.to_csv('data/svm_output.csv', index=False)