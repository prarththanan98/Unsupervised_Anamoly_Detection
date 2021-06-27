#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import tree

#%%
df=pd.read_csv('data/svm_output.csv', low_memory=False)
df.head()
df.info()
#%%
X=df.drop(['corona_result','model_prediction'] ,axis=1)
y=df['model_prediction']

#%%
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
importance=dtree.feature_importances_
print(importance)
for i,v in enumerate(importance):
    print('Feature: %s - score: %.5f'  %(df.columns[i],v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()
#%%
import graphviz
target=['positive','negative']
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=X.columns.values,
                                class_names=target,
                                filled=True)

graph = graphviz.Source(dot_data, format="png")
graph
graph.view()

